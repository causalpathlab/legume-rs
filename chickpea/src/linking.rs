use crate::common::*;
use crate::input::{find_cis_peaks, GeneTss, PeakCoord};
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::sgvb::cavi_susie::{cavi_susie, CaviSusieParams};
use candle_util::sgvb::variant_tree::VariantTree;
use candle_util::sgvb::VariationalDistribution;
use matrix_util::traits::ConvertMatOps;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum LinkModel {
    Gaussian,
    Poisson,
    #[value(alias = "nb")]
    NegBinom,
}

#[derive(Debug)]
pub struct GeneLinkResult {
    pub gene_idx: usize,
    pub gene_name: Box<str>,
    pub peak_indices: Vec<usize>,
    pub pip: Vec<f64>,
    pub effect_size: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CascadeParams {
    pub num_components: usize,
    pub prior_var: f64,
    pub prune_threshold: f64,
    pub pip_threshold: f64,
    pub sgvb: SgvbParams,
}

#[derive(Debug, Clone)]
pub struct SgvbParams {
    pub learning_rate: f64,
    pub num_iterations: usize,
    pub num_samples: usize,
}

impl Default for SgvbParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            num_iterations: 2000,
            num_samples: 4,
        }
    }
}

pub struct CascadeTask {
    /// [n, k] — k=1 for single gene, k>1 for gene block.
    pub y: Mat,
    pub candidate_peaks: Vec<usize>,
    pub gene_idx: usize,
    pub gene_name: Box<str>,
}

pub struct CascadeResult {
    pub peak_indices: Vec<usize>,
    pub pip: Vec<f64>,
    pub effect_size: Vec<f64>,
}

pub struct ModuleLinkScreen {
    pub corr_matrix: Mat,
    pub candidate_pairs: Vec<(usize, usize)>,
}

/// Precomputed tree-level mappings for efficient cascade traversal.
pub struct TreeLevelMaps {
    /// For each level: child_group → parent_group.
    pub child_to_parent: Vec<HashMap<usize, usize>>,
    /// For each level: parent_group → number of children.
    pub parent_child_count: Vec<HashMap<usize, usize>>,
}

impl TreeLevelMaps {
    pub fn from_tree(tree: &VariantTree) -> Self {
        let d = tree.num_variants;
        let depth = tree.depth;

        let mut child_to_parent = Vec::with_capacity(depth);
        let mut parent_child_count = Vec::with_capacity(depth);

        for level in 0..depth {
            let tree_level = &tree.levels[level];
            let mc = tree_level.max_children;

            if level == 0 {
                child_to_parent.push(Default::default());
                parent_child_count.push(Default::default());
                continue;
            }

            let prev = &tree.levels[level - 1];
            let prev_mc = prev.max_children;

            let mut c2p: HashMap<usize, usize> = Default::default();
            let mut pcc: HashMap<usize, usize> = Default::default();

            for j in 0..d {
                let child_g = tree_level.flat_path_indices[j] / mc;
                let parent_g = prev.flat_path_indices[j] / prev_mc;
                if c2p.insert(child_g, parent_g).is_none() {
                    *pcc.entry(parent_g).or_insert(0) += 1;
                }
            }

            child_to_parent.push(c2p);
            parent_child_count.push(pcc);
        }

        Self {
            child_to_parent,
            parent_child_count,
        }
    }
}

pub(crate) struct LevelState {
    pip: Vec<f64>,
    groups: Vec<usize>,
}

pub fn screen_module_links(
    gene_pb_coarse: &Mat,
    peak_pb_coarse: &Mat,
    threshold: f32,
) -> ModuleLinkScreen {
    let g = gene_pb_coarse.nrows();
    let p = peak_pb_coarse.nrows();
    let s = gene_pb_coarse.ncols();
    assert_eq!(s, peak_pb_coarse.ncols(), "Super-cell counts must match");

    let gene_z = gene_pb_coarse.map(|x| (x + 1.0).ln()).scale_rows();
    let peak_z = peak_pb_coarse.map(|x| (x + 1.0).ln()).scale_rows();

    let corr_matrix = &gene_z * peak_z.transpose() / s as f32;

    let mut candidate_pairs: Vec<(usize, usize)> = Vec::new();
    for gi in 0..g {
        for pi in 0..p {
            if corr_matrix[(gi, pi)].abs() > threshold {
                candidate_pairs.push((gi, pi));
            }
        }
    }

    info!(
        "Module screening: {} gene × {} peak modules, {} candidates (|r| > {:.2})",
        g,
        p,
        candidate_pairs.len(),
        threshold
    );

    ModuleLinkScreen {
        corr_matrix,
        candidate_pairs,
    }
}

pub fn expand_module_candidates(
    gene_coarsening: &FeatureCoarsening,
    peak_coarsening: &FeatureCoarsening,
    screen: &ModuleLinkScreen,
) -> Vec<Vec<usize>> {
    let num_genes = gene_coarsening.fine_to_coarse.len();
    let mut gene_candidates: Vec<Vec<usize>> = vec![Vec::new(); num_genes];

    let mut module_links: HashMap<usize, Vec<usize>> = Default::default();
    for &(gm, pm) in &screen.candidate_pairs {
        module_links.entry(gm).or_default().push(pm);
    }

    for (gene_idx, &gene_module) in gene_coarsening.fine_to_coarse.iter().enumerate() {
        if let Some(peak_modules) = module_links.get(&gene_module) {
            let total: usize = peak_modules
                .iter()
                .map(|&pm| peak_coarsening.coarse_to_fine[pm].len())
                .sum();
            let mut peaks = Vec::with_capacity(total);
            for &pm in peak_modules {
                peaks.extend_from_slice(&peak_coarsening.coarse_to_fine[pm]);
            }
            peaks.sort_unstable();
            peaks.dedup();
            gene_candidates[gene_idx] = peaks;
        }
    }

    gene_candidates
}

pub fn cis_filter_candidates(
    candidates: &[usize],
    gene_tss: Option<&GeneTss>,
    peak_coords: &[Option<PeakCoord>],
    cis_window: i64,
) -> Vec<usize> {
    match gene_tss {
        Some(tss) => {
            let cis_set: HashSet<usize> = find_cis_peaks(tss, peak_coords, cis_window)
                .into_iter()
                .collect();
            candidates
                .iter()
                .filter(|&&idx| cis_set.contains(&idx))
                .copied()
                .collect()
        }
        None => candidates.to_vec(),
    }
}

/// Aggregate module-level pseudobulk by tree grouping at a given level.
pub fn aggregate_modules_by_tree_level(
    module_pb: &Mat,
    tree: &VariantTree,
    level: usize,
    active_groups: Option<&[usize]>,
) -> (Mat, Vec<usize>) {
    let n = module_pb.ncols();
    let d = module_pb.nrows();
    let tree_level = &tree.levels[level];
    let mc = tree_level.max_children;
    let num_groups = tree_level.num_groups;

    let mut group_to_out: Vec<Option<usize>> = vec![None; num_groups];
    let mut out_groups: Vec<usize> = Vec::new();

    match active_groups {
        Some(ag) => {
            let active_set: HashSet<usize> = ag.iter().copied().collect();
            for (g, slot) in group_to_out.iter_mut().enumerate() {
                if active_set.contains(&g) {
                    *slot = Some(out_groups.len());
                    out_groups.push(g);
                }
            }
        }
        None => {
            for (g, slot) in group_to_out.iter_mut().enumerate() {
                *slot = Some(g);
                out_groups.push(g);
            }
        }
    }

    let n_out = out_groups.len();
    let mut result = Mat::zeros(n_out, n);

    for j in 0..d {
        let group = tree_level.flat_path_indices[j] / mc;
        if let Some(out_idx) = group_to_out.get(group).and_then(|x| *x) {
            for s in 0..n {
                result[(out_idx, s)] += module_pb[(j, s)];
            }
        }
    }

    (result, out_groups)
}

/// Build child prior weights from parent PIPs (hard mask + soft prior).
pub fn build_child_prior(
    parent_state: &LevelState,
    tree_maps: &TreeLevelMaps,
    level: usize,
    prune_threshold: f64,
) -> (Vec<f64>, Vec<usize>) {
    let c2p = &tree_maps.child_to_parent[level];
    let pcc = &tree_maps.parent_child_count[level];

    let parent_pip_map: HashMap<usize, f64> = parent_state
        .groups
        .iter()
        .zip(parent_state.pip.iter())
        .filter(|(_, &pip)| pip >= prune_threshold)
        .map(|(&g, &p)| (g, p))
        .collect();

    if parent_pip_map.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut child_groups_and_weights: Vec<(usize, f64)> = Vec::new();
    let tree_level_num_groups = c2p.keys().max().map_or(0, |&m| m + 1);

    for g in 0..tree_level_num_groups {
        if let Some(&pg) = c2p.get(&g) {
            if let Some(&parent_p) = parent_pip_map.get(&pg) {
                let n_children = pcc.get(&pg).copied().unwrap_or(1) as f64;
                child_groups_and_weights.push((g, parent_p / n_children));
            }
        }
    }

    if child_groups_and_weights.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let sum: f64 = child_groups_and_weights.iter().map(|(_, w)| w).sum();
    let active_groups: Vec<usize> = child_groups_and_weights.iter().map(|(g, _)| *g).collect();
    let weights: Vec<f64> = child_groups_and_weights
        .iter()
        .map(|(_, w)| w / sum)
        .collect();

    (weights, active_groups)
}

/// Build prior weights for fine-level peaks from parent module PIPs.
fn build_fine_prior(
    parent_state: &LevelState,
    fine_peaks: &[usize],
    fc: &FeatureCoarsening,
) -> Vec<f64> {
    let pip_map: HashMap<usize, f64> = parent_state
        .groups
        .iter()
        .zip(parent_state.pip.iter())
        .map(|(&g, &p)| (g, p))
        .collect();

    let mut module_peak_count: HashMap<usize, usize> = Default::default();
    for &pi in fine_peaks {
        *module_peak_count.entry(fc.fine_to_coarse[pi]).or_insert(0) += 1;
    }

    let mut weights: Vec<f64> = fine_peaks
        .iter()
        .map(|&pi| {
            let m = fc.fine_to_coarse[pi];
            let parent_p = pip_map.get(&m).copied().unwrap_or(1e-10);
            let n_in = module_peak_count.get(&m).copied().unwrap_or(1) as f64;
            parent_p / n_in
        })
        .collect();

    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    weights
}

/// Convert a Mat [rows, cols] to Tensor [cols, rows] with an element-wise transform.
fn mat_to_tensor_t(x: &Mat, transform: impl Fn(f32) -> f32) -> Option<Tensor> {
    let t = x.map(transform).to_tensor(&Device::Cpu).ok()?;
    t.transpose(0, 1).ok()
}

/// Extract a flattened f64 Vec from a Tensor.
fn tensor_to_f64_vec(t: &Tensor) -> Option<Vec<f64>> {
    Some(
        t.flatten_all()
            .ok()?
            .to_vec1::<f32>()
            .ok()?
            .iter()
            .map(|&x| x as f64)
            .collect(),
    )
}

/// Unified SuSiE dispatch for one level of variable selection.
fn run_susie_level(
    x: &Mat,
    y: &Mat,
    model: LinkModel,
    params: &CascadeParams,
    prior_weights: Option<Vec<f64>>,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if x.nrows() == 0 {
        return None;
    }
    match model {
        LinkModel::Gaussian => run_susie_cavi(x, y, params, prior_weights),
        LinkModel::Poisson | LinkModel::NegBinom => {
            run_susie_sgvb(x, y, model, params, prior_weights)
        }
    }
}

fn run_susie_cavi(
    x: &Mat,
    y: &Mat,
    params: &CascadeParams,
    prior_weights: Option<Vec<f64>>,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let x_t = mat_to_tensor_t(x, |v| (v + 1.0).ln())?;
    let y_t = mat_to_tensor_t(y, |v| (v + 1.0).ln())?;

    let susie_params = CaviSusieParams {
        num_components: params.num_components,
        prior_variance: params.prior_var,
        prior_weights,
        ..Default::default()
    };

    let result = cavi_susie(&x_t, &y_t, &susie_params).ok()?;
    let effect_size = result.beta_mean();
    Some((result.pip, effect_size))
}

fn run_susie_sgvb(
    x: &Mat,
    y: &Mat,
    _model: LinkModel,
    params: &CascadeParams,
    _prior_weights: Option<Vec<f64>>,
) -> Option<(Vec<f64>, Vec<f64>)> {
    use candle_util::candle_nn::{Optimizer, VarBuilder, VarMap};
    use candle_util::sgvb::{
        local_reparam_loss, GaussianPrior, RegressionSGVB, SGVBConfig, SusieVar,
    };

    let p = x.nrows();
    let n = x.ncols();
    let k = y.nrows();
    let device = Device::Cpu;

    let x_tensor = mat_to_tensor_t(x, |v| (v + 1.0).ln())?;
    let y_tensor = mat_to_tensor_t(y, |v| v)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let susie = SusieVar::new(vb.pp("susie"), params.num_components, p, k).ok()?;
    let prior = GaussianPrior::new(vb.pp("prior"), params.prior_var as f32).ok()?;
    let config = SGVBConfig::new(params.sgvb.num_samples);
    let sgvb_model = RegressionSGVB::from_variational(susie, x_tensor, prior, config);

    let y_mean: f32 = (0..n)
        .map(|s| (0..k).map(|ki| y[(ki, s)]).sum::<f32>())
        .sum::<f32>()
        / (n * k) as f32;
    let offset = y_mean.max(1e-4).ln();

    // TODO: dispatch on _model for NegBinom vs Poisson likelihood
    let likelihood = candle_util::sgvb::OffsetPoissonLikelihood::new(y_tensor, offset).ok()?;

    let mut optimizer =
        candle_util::candle_nn::AdamW::new_lr(varmap.all_vars(), params.sgvb.learning_rate).ok()?;

    for _ in 0..params.sgvb.num_iterations {
        let loss =
            local_reparam_loss(&sgvb_model, &likelihood, params.sgvb.num_samples, 1.0).ok()?;
        optimizer.backward_step(&loss).ok()?;
    }

    let pip = tensor_to_f64_vec(&sgvb_model.variational.pip().ok()?)?;
    let effect_size = tensor_to_f64_vec(&sgvb_model.variational.mean().ok()?)?;

    Some((pip, effect_size))
}

/// Precomputed feature hierarchy for the cascade.
pub struct FeatureHierarchy<'a> {
    pub fc: &'a FeatureCoarsening,
    pub tree: &'a VariantTree,
    pub tree_maps: &'a TreeLevelMaps,
}

/// Run the multi-resolution cascade for one task.
pub fn run_cascade(
    task: &CascadeTask,
    module_pb: &Mat,
    peak_pb: &Mat,
    hierarchy: &FeatureHierarchy<'_>,
    model: LinkModel,
    params: &CascadeParams,
) -> Option<CascadeResult> {
    if task.candidate_peaks.is_empty() {
        return None;
    }

    let FeatureHierarchy {
        fc,
        tree,
        tree_maps,
    } = hierarchy;
    let n = module_pb.ncols();
    let depth = tree.depth;
    let mut prev_state: Option<LevelState> = None;

    for level in 0..depth {
        let (active_groups, prior_weights) = if level == 0 {
            (None, None)
        } else {
            let state = prev_state.as_ref()?;
            let (weights, active) =
                build_child_prior(state, tree_maps, level, params.prune_threshold);
            if active.is_empty() {
                return None;
            }
            (Some(active), Some(weights))
        };

        let (x_level, groups_at_level) =
            aggregate_modules_by_tree_level(module_pb, tree, level, active_groups.as_deref());

        if groups_at_level.is_empty() {
            return None;
        }

        let (pip, _effect) = run_susie_level(&x_level, &task.y, model, params, prior_weights)?;

        info!(
            "  Level {}: {} groups, max PIP {:.3}",
            level,
            groups_at_level.len(),
            pip.iter().cloned().fold(0.0f64, f64::max)
        );

        prev_state = Some(LevelState {
            pip,
            groups: groups_at_level,
        });
    }

    // Expand surviving modules to individual peaks
    let state = prev_state?;
    let surviving_modules: Vec<usize> = state
        .groups
        .iter()
        .zip(state.pip.iter())
        .filter(|(_, &p)| p >= params.prune_threshold)
        .map(|(&g, _)| g)
        .collect();

    if surviving_modules.is_empty() {
        return None;
    }

    // Use binary search on sorted candidate_peaks instead of HashSet
    let mut fine_peaks: Vec<usize> = Vec::new();
    for &m in &surviving_modules {
        for &pi in &fc.coarse_to_fine[m] {
            if task.candidate_peaks.binary_search(&pi).is_ok() {
                fine_peaks.push(pi);
            }
        }
    }
    fine_peaks.sort_unstable();
    fine_peaks.dedup();

    if fine_peaks.is_empty() {
        return None;
    }

    let fine_prior = build_fine_prior(&state, &fine_peaks, fc);

    let p_fine = fine_peaks.len();
    let mut x_fine = Mat::zeros(p_fine, n);
    for (out_i, &pi) in fine_peaks.iter().enumerate() {
        x_fine.row_mut(out_i).copy_from(&peak_pb.row(pi));
    }

    info!(
        "  Fine level: {} peaks from {} surviving modules",
        p_fine,
        surviving_modules.len()
    );

    let (pip, effect_size) = run_susie_level(&x_fine, &task.y, model, params, Some(fine_prior))?;

    if !pip.iter().any(|&p| p >= params.pip_threshold) {
        return None;
    }

    Some(CascadeResult {
        peak_indices: fine_peaks,
        pip,
        effect_size,
    })
}

#[cfg(test)]
mod sim {
    use super::*;

    pub struct SimLinkParams {
        pub n_cells: usize,
        pub n_peaks: usize,
        pub n_genes: usize,
        pub n_causal: usize,
        pub heritability: f32,
        pub seed: u64,
    }

    #[allow(dead_code)]
    pub struct SimLinkData {
        pub peak_pb: Mat,
        pub gene_pb: Mat,
        pub causal_peaks: Vec<Vec<usize>>,
        pub effect_sizes: Vec<Vec<f32>>,
    }

    pub fn simulate_link_data(params: &SimLinkParams) -> SimLinkData {
        use rand::prelude::*;
        use rand_distr::{Gamma, Normal, Poisson};

        let n = params.n_cells;
        let d = params.n_peaks;
        let g = params.n_genes;
        let mut rng = StdRng::seed_from_u64(params.seed);

        let gamma = Gamma::new(2.0f64, 1.0).unwrap();
        let mut peak_pb = Mat::zeros(d, n);
        for i in 0..d {
            let lambda: f64 = gamma.sample(&mut rng);
            let pois = Poisson::new(lambda.max(0.1)).unwrap();
            for j in 0..n {
                peak_pb[(i, j)] = pois.sample(&mut rng) as f32;
            }
        }

        let peak_log = peak_pb.map(|x| (x + 1.0).ln());
        let normal = Normal::new(0.0f64, 1.0).unwrap();

        let mut gene_pb = Mat::zeros(g, n);
        let mut causal_peaks = Vec::with_capacity(g);
        let mut effect_sizes = Vec::with_capacity(g);

        for gi in 0..g {
            let gene_seed = params.seed.wrapping_add(gi as u64);
            let mut gene_rng = StdRng::seed_from_u64(gene_seed);

            let nc = params.n_causal.min(d);
            let causal_idx: Vec<usize> = rand::seq::index::sample(&mut gene_rng, d, nc).into_vec();

            let effect_std = 1.0 / (nc as f64).sqrt();
            let effects: Vec<f32> = (0..nc)
                .map(|_| (normal.sample(&mut gene_rng) * effect_std) as f32)
                .collect();

            let mut signal = vec![0.0f32; n];
            for (ci, &pi) in causal_idx.iter().enumerate() {
                for s in 0..n {
                    signal[s] += peak_log[(pi, s)] * effects[ci];
                }
            }

            standardize_vec(&mut signal);

            let mut noise: Vec<f32> = (0..n)
                .map(|_| normal.sample(&mut gene_rng) as f32)
                .collect();
            standardize_vec(&mut noise);

            let h_sig = params.heritability.sqrt();
            let h_noise = (1.0 - params.heritability).sqrt();
            for s in 0..n {
                gene_pb[(gi, s)] = (h_sig * signal[s] + h_noise * noise[s]).exp();
            }

            causal_peaks.push(causal_idx);
            effect_sizes.push(effects);
        }

        SimLinkData {
            peak_pb,
            gene_pb,
            causal_peaks,
            effect_sizes,
        }
    }

    fn standardize_vec(v: &mut [f32]) {
        let n = v.len() as f32;
        let mean: f32 = v.iter().sum::<f32>() / n;
        let var: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = var.sqrt().max(1e-8);
        for x in v.iter_mut() {
            *x = (*x - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sim::*;
    use super::*;

    #[test]
    fn test_aggregate_modules_by_tree_level() {
        let tree = VariantTree::regular(8, 2);
        assert!(tree.depth >= 2);

        let mut module_pb = Mat::zeros(8, 4);
        for i in 0..8 {
            for j in 0..4 {
                module_pb[(i, j)] = (i * 4 + j) as f32;
            }
        }

        let (agg, groups) = aggregate_modules_by_tree_level(&module_pb, &tree, 0, None);
        assert!(agg.nrows() <= 8);
        assert_eq!(agg.ncols(), 4);
        assert_eq!(agg.nrows(), groups.len());

        for s in 0..4 {
            let agg_sum: f32 = (0..agg.nrows()).map(|g| agg[(g, s)]).sum();
            let orig_sum: f32 = (0..8).map(|m| module_pb[(m, s)]).sum();
            assert!(
                (agg_sum - orig_sum).abs() < 1e-4,
                "Sample {}: agg_sum {} != orig_sum {}",
                s,
                agg_sum,
                orig_sum
            );
        }

        if groups.len() >= 2 {
            let active = vec![groups[0]];
            let (agg2, groups2) =
                aggregate_modules_by_tree_level(&module_pb, &tree, 0, Some(&active));
            assert_eq!(groups2.len(), 1);
            assert_eq!(agg2.nrows(), 1);
        }
    }

    #[test]
    fn test_build_child_prior() {
        let tree = VariantTree::regular(16, 4);
        let tree_maps = TreeLevelMaps::from_tree(&tree);
        assert!(tree.depth >= 2);

        let (_, groups_0) = aggregate_modules_by_tree_level(&Mat::zeros(16, 1), &tree, 0, None);

        let mut parent_pip = vec![0.01f64; groups_0.len()];
        parent_pip[0] = 0.9;

        let parent_state = LevelState {
            pip: parent_pip,
            groups: groups_0,
        };

        let (weights, active) = build_child_prior(&parent_state, &tree_maps, 1, 0.05);
        assert!(!active.is_empty());
        assert!(!weights.is_empty());

        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Prior weights sum to {}", sum);
    }

    #[test]
    fn test_run_susie_level_gaussian() {
        use rand::prelude::*;
        use rand_distr::Poisson;

        let n = 200;
        let p = 10;
        let causal = 3;
        let mut rng = StdRng::seed_from_u64(42);

        let mut x = Mat::zeros(p, n);
        for i in 0..p {
            let lambda = 5.0 + (i as f64) * 2.0;
            let pois = Poisson::new(lambda).unwrap();
            for j in 0..n {
                x[(i, j)] = pois.sample(&mut rng) as f32;
            }
        }

        let mut y = Mat::zeros(1, n);
        for s in 0..n {
            let signal = 2.0 * (x[(causal, s)] + 1.0).ln();
            let noise: f32 = rand_distr::Normal::new(0.0f32, 0.3)
                .unwrap()
                .sample(&mut rng);
            y[(0, s)] = (signal + noise).exp();
        }

        let params = CascadeParams {
            num_components: 3,
            prior_var: 1.0,
            prune_threshold: 0.05,
            pip_threshold: 0.1,
            sgvb: SgvbParams::default(),
        };

        let (pip, _effect) = run_susie_level(&x, &y, LinkModel::Gaussian, &params, None).unwrap();
        assert_eq!(pip.len(), p);

        let max_pip_idx = pip
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(
            max_pip_idx, causal,
            "Expected causal idx {}, got {}. PIPs: {:?}",
            causal, max_pip_idx, pip
        );
    }

    #[test]
    fn test_cascade_produces_output() {
        let sim = simulate_link_data(&SimLinkParams {
            n_cells: 300,
            n_peaks: 50,
            n_genes: 1,
            n_causal: 1,
            heritability: 0.9,
            seed: 12345,
        });

        let fc = compute_feature_coarsening(&sim.peak_pb, 16).unwrap();
        let tree = VariantTree::regular(fc.num_coarse, 4);
        let tree_maps = TreeLevelMaps::from_tree(&tree);
        let module_pb = fc.aggregate_rows_ds(&sim.peak_pb);

        let task = CascadeTask {
            y: sim.gene_pb.rows(0, 1).into(),
            candidate_peaks: (0..50).collect(),
            gene_idx: 0,
            gene_name: "gene_0".into(),
        };

        let params = CascadeParams {
            num_components: 5,
            prior_var: 1.0,
            prune_threshold: 0.005,
            pip_threshold: 0.1,
            sgvb: SgvbParams::default(),
        };

        let hierarchy = FeatureHierarchy {
            fc: &fc,
            tree: &tree,
            tree_maps: &tree_maps,
        };
        let result = run_cascade(
            &task,
            &module_pb,
            &sim.peak_pb,
            &hierarchy,
            LinkModel::Gaussian,
            &params,
        );

        assert!(result.is_some(), "Cascade should find links for h²=0.9");
        let res = result.unwrap();
        assert!(res.peak_indices.len() < 50);
        assert!(res.pip.iter().any(|&p| p >= 0.1));
        for &p in &res.pip {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn test_cascade_prunes_effectively() {
        let sim = simulate_link_data(&SimLinkParams {
            n_cells: 200,
            n_peaks: 2000,
            n_genes: 1,
            n_causal: 3,
            heritability: 0.5,
            seed: 54321,
        });

        let fc = compute_feature_coarsening(&sim.peak_pb, 128).unwrap();
        let tree = VariantTree::regular(fc.num_coarse, 4);
        let tree_maps = TreeLevelMaps::from_tree(&tree);
        let module_pb = fc.aggregate_rows_ds(&sim.peak_pb);

        let task = CascadeTask {
            y: sim.gene_pb.rows(0, 1).into(),
            candidate_peaks: (0..2000).collect(),
            gene_idx: 0,
            gene_name: "gene_0".into(),
        };

        let params = CascadeParams {
            num_components: 5,
            prior_var: 0.5,
            prune_threshold: 0.02,
            pip_threshold: 0.1,
            sgvb: SgvbParams::default(),
        };

        let hierarchy = FeatureHierarchy {
            fc: &fc,
            tree: &tree,
            tree_maps: &tree_maps,
        };
        let result = run_cascade(
            &task,
            &module_pb,
            &sim.peak_pb,
            &hierarchy,
            LinkModel::Gaussian,
            &params,
        );

        if let Some(res) = result {
            assert!(res.peak_indices.len() < 500);
        }
    }

    #[test]
    fn test_cascade_no_signal() {
        let sim = simulate_link_data(&SimLinkParams {
            n_cells: 200,
            n_peaks: 200,
            n_genes: 1,
            n_causal: 2,
            heritability: 0.0,
            seed: 99999,
        });

        let fc = compute_feature_coarsening(&sim.peak_pb, 32).unwrap();
        let tree = VariantTree::regular(fc.num_coarse, 4);
        let tree_maps = TreeLevelMaps::from_tree(&tree);
        let module_pb = fc.aggregate_rows_ds(&sim.peak_pb);

        let task = CascadeTask {
            y: sim.gene_pb.rows(0, 1).into(),
            candidate_peaks: (0..200).collect(),
            gene_idx: 0,
            gene_name: "gene_0".into(),
        };

        let params = CascadeParams {
            num_components: 5,
            prior_var: 0.5,
            prune_threshold: 0.1,
            pip_threshold: 0.9,
            sgvb: SgvbParams::default(),
        };

        let hierarchy = FeatureHierarchy {
            fc: &fc,
            tree: &tree,
            tree_maps: &tree_maps,
        };
        let result = run_cascade(
            &task,
            &module_pb,
            &sim.peak_pb,
            &hierarchy,
            LinkModel::Gaussian,
            &params,
        );

        if let Some(ref res) = result {
            let high_pip = res.pip.iter().filter(|&&p| p >= 0.9).count();
            assert!(high_pip == 0);
        }
    }

    #[test]
    fn test_cascade_multi_gene_parallel() {
        use rayon::prelude::*;

        let sim = simulate_link_data(&SimLinkParams {
            n_cells: 200,
            n_peaks: 500,
            n_genes: 5,
            n_causal: 2,
            heritability: 0.5,
            seed: 77777,
        });

        let fc = compute_feature_coarsening(&sim.peak_pb, 64).unwrap();
        let tree = VariantTree::regular(fc.num_coarse, 4);
        let tree_maps = TreeLevelMaps::from_tree(&tree);
        let module_pb = fc.aggregate_rows_ds(&sim.peak_pb);

        let params = CascadeParams {
            num_components: 5,
            prior_var: 0.5,
            prune_threshold: 0.01,
            pip_threshold: 0.1,
            sgvb: SgvbParams::default(),
        };

        let tasks: Vec<CascadeTask> = (0..5)
            .map(|gi| CascadeTask {
                y: sim.gene_pb.rows(gi, 1).into(),
                candidate_peaks: (0..500).collect(),
                gene_idx: gi,
                gene_name: format!("gene_{}", gi).into(),
            })
            .collect();

        let hierarchy = FeatureHierarchy {
            fc: &fc,
            tree: &tree,
            tree_maps: &tree_maps,
        };
        let results: Vec<Option<CascadeResult>> = tasks
            .par_iter()
            .map(|task| {
                run_cascade(
                    task,
                    &module_pb,
                    &sim.peak_pb,
                    &hierarchy,
                    LinkModel::Gaussian,
                    &params,
                )
            })
            .collect();

        assert!(results.iter().filter(|r| r.is_some()).count() > 0);
    }

    #[test]
    fn test_simulate_link_data_deterministic() {
        let sim1 = simulate_link_data(&SimLinkParams {
            n_cells: 50,
            n_peaks: 100,
            n_genes: 3,
            n_causal: 2,
            heritability: 0.5,
            seed: 42,
        });
        let sim2 = simulate_link_data(&SimLinkParams {
            n_cells: 50,
            n_peaks: 100,
            n_genes: 3,
            n_causal: 2,
            heritability: 0.5,
            seed: 42,
        });

        assert_eq!(sim1.causal_peaks, sim2.causal_peaks);
        for i in 0..100 {
            for j in 0..50 {
                assert_eq!(sim1.peak_pb[(i, j)], sim2.peak_pb[(i, j)]);
            }
        }
    }
}
