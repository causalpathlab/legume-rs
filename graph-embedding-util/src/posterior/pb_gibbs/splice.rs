//! Two-gate (splice) variant — `faba gem`. One `β_g` per gene plus a velocity deviation
//! `δ_g` carried only by the gene's unspliced rows, sampled as THREE blocks (`β | δ, pb`,
//! `δ | β, pb`, `pb | β, δ`) where the plain path has two. The extra gate brings its own
//! identifiability verdict per gene and its own accumulator over the outer sweeps.

use super::shared::{
    block_seed, profiled_bias, report_convergence, run_block, scatter_splice_bias_to_rows,
    scatter_splice_to_rows, warm_start_genes,
};
use super::{BetaTerm, PartitionGeometry, PbGibbsConfig};
use crate::fit::stacked_pb::StackedPb;
use crate::posterior::diagnostics::{scalar_diagnostics, ChainDiag};
use crate::posterior::dim_block::{dim_block, dim_block_multi, DimBlockConfig, DimBlockResult};
use crate::posterior::lnpdf::{FrozenSide, NodeTerm};
use crate::posterior::pb_index::{build_pb_index_pair, AnchorMap, FeatureSide};
use crate::progress::new_progress_bar;
use log::info;

/// `faba gem`'s β-sharing feature side: one `β_g` per gene, plus a velocity
/// deviation `δ_g` that applies only to the gene's **unspliced** rows.
pub struct SpliceTracks<'a> {
    /// Feature row → gene, length `n_features`; `u32::MAX` drops the row.
    pub row_to_gene: &'a [u32],
    /// Which rows carry `δ_g` (an unspliced row scores `⟨β_g + δ_g, e_p⟩`).
    pub unspliced_rows: &'a [bool],
    pub n_genes: usize,
    /// Allow `z_δ = 1` only where `z_β = 1`. See [`DimBlockConfig::z_allowed`] for
    /// why this is the default.
    pub nested: bool,
}

/// Both gates' posteriors, plus which genes could identify `δ` at all.
pub struct SpliceGibbsResult {
    pub beta_pip: Vec<f32>,
    pub beta_mean: Vec<f32>,
    pub delta_pip: Vec<f32>,
    pub delta_mean: Vec<f32>,
    /// Per gene: does it have SPLICED counts? `δ_g` is identified by the
    /// unspliced-minus-spliced contrast, so a gene observed only in the unspliced
    /// track pins `β + δ` but neither separately — its `δ` row is a prior draw,
    /// not a measurement, and must be reported as such rather than as a number.
    pub delta_identified: Vec<bool>,
    pub mean_pb: Vec<f32>,
    /// Per-gene and per-pseudobulk intercepts implied by the posterior-mean
    /// loadings — see [`super::PbGibbsResult::mean_b_feat`] for why they must travel
    /// together with them.
    pub mean_b_feat: Vec<f32>,
    pub mean_b_pb: Vec<f32>,
    pub beta_sigma2: Vec<f64>,
    pub beta_pi0: Vec<f64>,
    pub delta_sigma2: Vec<f64>,
    pub delta_pi0: Vec<f64>,
    /// Mixing diagnostics per dim, over the OUTER sweeps — see `SpliceAcc`.
    /// What the normalizers actually summed over — see [`PartitionGeometry`].
    pub partition: PartitionGeometry,
    pub beta_sigma_diag: Vec<ChainDiag>,
    pub beta_pi0_diag: Vec<ChainDiag>,
    pub delta_sigma_diag: Vec<ChainDiag>,
    pub delta_pi0_diag: Vec<ChainDiag>,
    pub h: usize,
    pub n_kept: usize,
}

/// Alternating sweep over gem's three blocks: `β | δ, pb`, then `δ | β, pb`, then
/// `pb | β, δ`.
///
/// **`β` reads the spliced rows only.** A [`NodeTerm`] carries one offset for all
/// of an anchor's edges, so "β with δ added on just the unspliced edges" is not
/// expressible as a single node; splitting β's likelihood across the two tracks
/// would need a per-edge offset. Reading β off the spliced rows is also the
/// cleaner estimand — those rows score exactly `⟨β_g, e_p⟩` with nothing to
/// disentangle — and it is what gem's previous posterior did.
///
/// `δ` then reads the unspliced rows with `β_g` as its per-anchor offset,
/// **refreshed every sweep**. Holding it at a one-time MAP snapshot, as the
/// previous implementation did, makes the two conditionals describe no single
/// joint.
pub(crate) fn pb_gibbs_splice(
    pb: &StackedPb<'_>,
    feat: &FeatureSide<'_>,
    tracks: &SpliceTracks<'_>,
    h: usize,
    cfg: &PbGibbsConfig,
) -> anyhow::Result<SpliceGibbsResult> {
    let n_genes = tracks.n_genes;
    anyhow::ensure!(
        tracks.row_to_gene.len() == feat.b_feat.len()
            && tracks.unspliced_rows.len() == feat.b_feat.len(),
        "splice tracks disagree with the {}-row feature axis",
        feat.b_feat.len()
    );

    // One anchor map per track: a row belongs to a track's gene only if it is on
    // that track. Everything else is dropped from THAT gene side (never from the
    // pb side — `build_pb_index_pair` keeps every row there by construction).
    let track_map = |want_unspliced: bool| -> Vec<u32> {
        tracks
            .row_to_gene
            .iter()
            .zip(tracks.unspliced_rows)
            .map(|(&g, &u)| if u == want_unspliced { g } else { u32::MAX })
            .collect()
    };
    let spliced_map = track_map(false);
    let unspliced_map = track_map(true);

    let beta_anchors = AnchorMap {
        row_to_anchor: &spliced_map,
        n_anchors: n_genes,
    };
    let delta_anchors = AnchorMap {
        row_to_anchor: &unspliced_map,
        n_anchors: n_genes,
    };
    let beta_pair =
        build_pb_index_pair(pb, feat, Some(&beta_anchors), h, cfg.n_partition, cfg.seed)?;
    let delta_pair =
        build_pb_index_pair(pb, feat, Some(&delta_anchors), h, cfg.n_partition, cfg.seed)?;

    // Identifiability, decided up front and reported, not inferred later.
    //
    // δ needs counts on BOTH tracks. No spliced counts ⇒ only β+δ is pinned. No
    // UNSPLICED counts ⇒ δ appears in no likelihood term at all: `multinomial_ll`
    // returns 0 for every argument via its `total == 0` early return, so `z` is
    // drawn straight from the prior odds and β_δ from the slab. Testing only the
    // spliced side would ship that second case as an apparently-measured number.
    let delta_identified: Vec<bool> = beta_pair
        .by_feature
        .pos
        .iter()
        .zip(&delta_pair.by_feature.pos)
        .map(|(spliced, unspliced)| !spliced.is_empty() && !unspliced.is_empty())
        .collect();
    let n_unidentified = delta_identified.iter().filter(|&&x| !x).count();
    if n_unidentified > 0 {
        log::warn!(
            "{n_unidentified} of {n_genes} gene(s) lack counts on one of the two splice \
             tracks, so their δ is not identified — with no spliced counts only β+δ is \
             pinned, and with no unspliced counts δ enters no likelihood term at all. \
             Both cases are flagged in `delta_identified` and written as NaN."
        );
    }
    // Per-TRACK edge counts, from the anchor-side lists. `PbIndexPair::n_edges` is
    // the whole matrix's observed edges — it is incremented once per nonzero
    // regardless of the anchor map — so reporting it here would print the same
    // number twice and claim it was the split.
    let n_spliced: usize = beta_pair.by_feature.pos.iter().map(Vec::len).sum();
    let n_unspliced: usize = delta_pair.by_feature.pos.iter().map(Vec::len).sum();
    info!(
        "gem splice Gibbs: {n_genes} gene(s), {} spliced / {} unspliced edge(s), {} gate \
         ({} sweeps +{} warmup); partitions {} pb / {} feature (scale {:.2} / {:.2})",
        n_spliced,
        n_unspliced,
        if tracks.nested {
            "nested z_δ ⊆ z_β"
        } else {
            "independent"
        },
        cfg.n_sweeps,
        cfg.burnin,
        beta_pair.by_feature.partition.len(),
        beta_pair.by_pb.partition.len(),
        beta_pair.by_feature.partition_scale,
        beta_pair.by_pb.partition_scale,
    );
    // Cost, up front, because this run is long enough that finding out empirically is
    // expensive. A sweep is THREE blocks and the gene side of it is THREE term-passes
    // (β over both tracks, δ over unspliced) where `senna bge`'s is one — so at equal
    // `h`, gene count and slate, gem's gene side is ~3x bge's. Measured on this tree at
    // `h=128`, slate 1460, saturating anchors: 87 ms per 1000 anchors per term-pass, and
    // the pass is LINEAR in `h` and in the slate, flat in edge count (the `AnchorMoment`
    // collapse is why). So the levers, in measured order of effect, are `--threads`
    // (16 by default, well under a big box), `--embedding-dim`, and the sweep count —
    // NOT the edge count, which is what a slow run tempts you to cut first.
    info!(
        "gem splice Gibbs cost ~ n_genes × h × slate × 3 term-passes per sweep: \
         {n_genes} × {h} × {} × 3 × {} sweeps, on {} rayon thread(s)",
        beta_pair.by_feature.partition.len(),
        cfg.n_sweeps + cfg.burnin,
        rayon::current_num_threads(),
    );

    let mut e_beta = warm_start_genes(feat, Some(&beta_anchors), n_genes, h);
    let mut e_delta = vec![0f32; n_genes * h];
    let mut e_pb = pb.theta.clone();
    let mut e_rows = feat.e_feat.to_vec();
    // Intercepts are LIVE here for the same reason as on the plain path: each block
    // profiles its own anchor's intercept out exactly, so the other side must score
    // against that rather than against a pre-sampling snapshot. See the plain path's
    // note on why this is a profile step and not a draw.
    let mut b_pb_live = beta_pair.by_feature.other_b.clone();
    let mut b_row_live = beta_pair.by_pb.other_b.clone();
    // Inclusion state carried between outer sweeps, one per gate — without it the
    // ESS branch is unreachable and every draw comes from the prior.
    let (mut z_beta, mut z_delta): (Option<Vec<bool>>, Option<Vec<bool>>) = (None, None);

    let mut acc = SpliceAcc::new(n_genes, pb.n_pb_total(), h);
    let (mut fallbacks, mut n_transitions) = (0usize, 0usize);
    let stop = crate::stop::stop_flag();
    let total = cfg.n_sweeps + cfg.burnin;
    let bar = new_progress_bar(total as u64).with_message("gem splice sweeps");

    for sweep in 0..total {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let keep = sweep >= cfg.burnin;
        let side_pb = FrozenSide {
            e: &e_pb,
            b: &b_pb_live,
            h,
        };

        // β | δ, pb — over BOTH tracks. β appears in the unspliced rows too
        // (they score `⟨β + δ, e_p⟩`), so a spliced-only kernel would not be the
        // conditional `p(β | δ, ·)` and the three blocks would stop being
        // conditionals of one joint. The unspliced term carries δ as its offset;
        // the two tracks have independent per-row biases, so each term's
        // intercept profiles out separately and the sum is the joint's profile.
        let beta_terms: Vec<Vec<NodeTerm>> = (0..n_genes)
            .map(|a| {
                let mut terms = vec![
                    NodeTerm::new(
                        &beta_pair.by_feature.pos[a],
                        &beta_pair.by_feature.partition,
                        beta_pair.by_feature.partition_scale,
                    );
                    BetaTerm::COUNT
                ];
                terms[BetaTerm::UNSPLICED] = {
                    let mut t = NodeTerm::new(
                        &delta_pair.by_feature.pos[a],
                        &delta_pair.by_feature.partition,
                        delta_pair.by_feature.partition_scale,
                    );
                    t.offset = Some(&e_delta[a * h..(a + 1) * h]);
                    t
                };
                terms
            })
            .collect();
        let mut beta_cfg = DimBlockConfig::new(1, 0, block_seed(cfg.seed, 0xB37A, sweep))
            .with_init_beta(e_beta.clone())
            .with_label("beta|delta,pb")
            .quiet();
        if let Some(z) = z_beta.clone() {
            beta_cfg = beta_cfg.with_init_z(z);
        }
        beta_cfg.transitions_per_dim = cfg.transitions_per_dim;
        beta_cfg.stick_alpha = cfg.stick_alpha;
        let b = dim_block_multi(&beta_terms, &side_pb, &beta_cfg);
        drop(beta_terms);
        e_beta.copy_from_slice(&b.mean_beta);
        z_beta = Some(b.final_z.clone());

        // δ | β, pb, on the unspliced rows, with β carried as the per-anchor
        // offset so the sampler explores a deviation rather than an absolute
        // loading — refreshed from THIS sweep's β.
        let d = run_block(
            &delta_pair.by_feature.pos,
            &delta_pair.by_feature.partition,
            delta_pair.by_feature.partition_scale,
            &side_pb,
            &e_delta,
            Some(&e_beta),
            tracks.nested.then(|| b.final_z.clone()),
            z_delta.clone(),
            cfg,
            sweep,
            0xD317,
            "delta|beta",
        );
        e_delta.copy_from_slice(&d.mean_beta);
        z_delta = Some(d.final_z.clone());
        // Row intercepts, each from the block that last moved its track. The β block
        // carries two terms (spliced at 0, unspliced at 1), but δ ran after it, so an
        // unspliced row's current intercept is δ's; spliced rows take β's term 0.
        scatter_splice_bias_to_rows(&b, &d, tracks, feat.b_feat, &mut b_row_live);

        // pb | β, δ. The pb side sees the EFFECTIVE per-row loading: β on spliced
        // rows, β+δ on unspliced ones — which is what the model scores.
        scatter_splice_to_rows(&e_beta, &e_delta, tracks, feat.e_feat, h, &mut e_rows);
        let side_rows = FrozenSide {
            e: &e_rows,
            b: &b_row_live,
            h,
        };
        let mut pb_cfg = DimBlockConfig::new(1, 0, block_seed(cfg.seed, 0x85EB, sweep))
            .with_init_beta(e_pb.clone())
            .with_label("pb|beta,delta")
            .without_selection()
            .quiet();
        pb_cfg.transitions_per_dim = cfg.transitions_per_dim;
        let pbres = {
            let nodes: Vec<NodeTerm> = beta_pair
                .by_pb
                .pos
                .iter()
                .map(|p| {
                    NodeTerm::new(
                        p,
                        &beta_pair.by_pb.partition,
                        beta_pair.by_pb.partition_scale,
                    )
                })
                .collect();
            dim_block(&nodes, &side_rows, &pb_cfg)
        };
        e_pb.copy_from_slice(&pbres.mean_beta);
        b_pb_live.copy_from_slice(&pbres.b_profiled);

        fallbacks += b.fallbacks + d.fallbacks + pbres.fallbacks;
        n_transitions += b.n_transitions + d.n_transitions + pbres.n_transitions;
        if keep {
            acc.add(&b, &d, &pbres);
        }
        bar.inc(1);
    }
    bar.finish_and_clear();
    let mut out = acc.finish(
        h,
        delta_identified,
        PartitionGeometry {
            pb_entries: beta_pair.by_feature.partition.len(),
            pb_scale: beta_pair.by_feature.partition_scale,
            feat_entries: beta_pair.by_pb.partition.len(),
            feat_scale: beta_pair.by_pb.partition_scale,
        },
    )?;
    report_convergence(
        "gem splice (β gate)",
        &out.beta_sigma_diag,
        &out.beta_pi0_diag,
        // Summed over all THREE blocks, so it belongs to the run, not to this gate.
        Some((fallbacks, n_transitions)),
    );
    // The δ gate gets its own line: it is a different object on a different scale, so a
    // shared summary would let one gate's healthy chains cover for the other's.
    report_convergence(
        "gem splice (δ gate)",
        &out.delta_sigma_diag,
        &out.delta_pi0_diag,
        None,
    );

    // Intercepts against the posterior-mean sides, same reasoning as the bge path.
    let side_pb_final = FrozenSide {
        e: &out.mean_pb,
        b: &b_pb_live,
        h,
    };
    out.mean_b_feat = (0..n_genes)
        .map(|g| {
            let total: f64 = beta_pair.by_feature.pos[g]
                .iter()
                .chain(&delta_pair.by_feature.pos[g])
                .map(|&(_, n)| f64::from(n))
                .sum();
            profiled_bias(
                total,
                &out.beta_mean[g * h..(g + 1) * h],
                None,
                &beta_pair.by_feature.partition,
                beta_pair.by_feature.partition_scale,
                &side_pb_final,
            )
        })
        .collect();
    scatter_splice_to_rows(
        &out.beta_mean,
        &out.delta_mean,
        tracks,
        feat.e_feat,
        h,
        &mut e_rows,
    );
    let side_rows_final = FrozenSide {
        e: &e_rows,
        b: &b_row_live,
        h,
    };
    out.mean_b_pb = (0..pb.n_pb_total())
        .map(|p| {
            let total: f64 = beta_pair.by_pb.pos[p]
                .iter()
                .map(|&(_, n)| f64::from(n))
                .sum();
            profiled_bias(
                total,
                &out.mean_pb[p * h..(p + 1) * h],
                None,
                &beta_pair.by_pb.partition,
                beta_pair.by_pb.partition_scale,
                &side_rows_final,
            )
        })
        .collect();
    Ok(out)
}

/// Running sums for the splice sweep; kept in one place so the three blocks'
/// accumulators cannot drift out of step.
struct SpliceAcc {
    bp: Vec<f64>,
    bm: Vec<f64>,
    dp: Vec<f64>,
    dm: Vec<f64>,
    pbm: Vec<f64>,
    /// Per-OUTER-sweep hyper chains, one vec per dim. Sums would be enough for
    /// the posterior means, but not for `scalar_diagnostics` — and the inner
    /// blocks run a single sweep each, so their own chains hold one sample and
    /// report ESS = 1 unconditionally.
    bs2: Vec<Vec<f64>>,
    bpi: Vec<Vec<f64>>,
    ds2: Vec<Vec<f64>>,
    dpi: Vec<Vec<f64>>,
    n: usize,
}

impl SpliceAcc {
    fn new(n_genes: usize, n_pb: usize, h: usize) -> Self {
        Self {
            bp: vec![0.0; n_genes * h],
            bm: vec![0.0; n_genes * h],
            dp: vec![0.0; n_genes * h],
            dm: vec![0.0; n_genes * h],
            pbm: vec![0.0; n_pb * h],
            bs2: vec![Vec::new(); h],
            bpi: vec![Vec::new(); h],
            ds2: vec![Vec::new(); h],
            dpi: vec![Vec::new(); h],
            n: 0,
        }
    }

    fn add(&mut self, b: &DimBlockResult, d: &DimBlockResult, pb: &DimBlockResult) {
        let acc = |dst: &mut [f64], src: &[f32]| {
            for (a, v) in dst.iter_mut().zip(src) {
                *a += f64::from(*v);
            }
        };
        let push64 = |dst: &mut [Vec<f64>], src: &[f64]| {
            for (a, v) in dst.iter_mut().zip(src) {
                a.push(*v);
            }
        };
        acc(&mut self.bp, &b.pip);
        acc(&mut self.bm, &b.mean_beta);
        acc(&mut self.dp, &d.pip);
        acc(&mut self.dm, &d.mean_beta);
        acc(&mut self.pbm, &pb.mean_beta);
        push64(&mut self.bs2, &b.sigma2);
        push64(&mut self.bpi, &b.pi0);
        push64(&mut self.ds2, &d.sigma2);
        push64(&mut self.dpi, &d.pi0);
        self.n += 1;
    }

    fn finish(
        self,
        h: usize,
        delta_identified: Vec<bool>,
        partition: PartitionGeometry,
    ) -> anyhow::Result<SpliceGibbsResult> {
        anyhow::ensure!(self.n > 0, "gem splice Gibbs retained zero sweeps");
        let inv = 1.0 / self.n as f64;
        let f32s = |v: &[f64]| -> Vec<f32> { v.iter().map(|&a| (a * inv) as f32).collect() };
        let f64s = |c: &[Vec<f64>]| -> Vec<f64> {
            c.iter()
                .map(|v| v.iter().sum::<f64>() / v.len().max(1) as f64)
                .collect()
        };
        let diag = |c: &[Vec<f64>]| -> Vec<ChainDiag> {
            c.iter().map(|v| scalar_diagnostics(v)).collect()
        };
        Ok(SpliceGibbsResult {
            beta_pip: f32s(&self.bp),
            beta_mean: f32s(&self.bm),
            delta_pip: f32s(&self.dp),
            delta_mean: f32s(&self.dm),
            delta_identified,
            mean_pb: f32s(&self.pbm),
            // Filled by the caller once the posterior means exist — they are
            // computed against those means, so they cannot be accumulated here.
            mean_b_feat: Vec::new(),
            mean_b_pb: Vec::new(),
            beta_sigma2: f64s(&self.bs2),
            beta_pi0: f64s(&self.bpi),
            delta_sigma2: f64s(&self.ds2),
            delta_pi0: f64s(&self.dpi),
            partition,
            beta_sigma_diag: diag(&self.bs2),
            beta_pi0_diag: diag(&self.bpi),
            delta_sigma_diag: diag(&self.ds2),
            delta_pi0_diag: diag(&self.dpi),
            h,
            n_kept: self.n,
        })
    }
}
