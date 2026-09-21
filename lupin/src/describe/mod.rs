//! `lupin describe` — compose a short sentence from annotate / lineage_annot evidence.
//!
//! The composer **never decides**: evidence comes from annotate artifacts (and optional
//! keyword incidence from a `word-graph` vocab). Sentences are citation-checked against
//! the allowed entity set. Candle Hub decoding is not wired yet — templates only.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use clap::Args;
use gene_text::vocab::Vocabulary;
use legume_numeric::matrix::parquet::peek_parquet_field_names;
use log::info;
use serde_json::json;

#[derive(Args, Debug)]
pub struct DescribeArgs {
    #[arg(
        long,
        short = 'f',
        help = "Annotate / lineage output prefix (reads `{from}.annot.parquet` or `{from}.argmax.tsv`)"
    )]
    pub from: Box<str>,

    #[arg(
        long,
        help = "Optional `word-graph` / `text-qc` prefix; fishes keywords from `{prefix}.feature_word.edges.tsv` via each cluster's markers"
    )]
    pub text_prefix: Option<Box<str>>,

    #[arg(
        long,
        help = "Optional marker TSV (`gene<TAB>celltype`) when annotate did not write marker tables"
    )]
    pub markers: Option<Box<str>>,

    #[arg(
        long,
        help = "Optional feature×H embedding parquet; expands each cluster's marker set by nearest neighbours before fishing keywords"
    )]
    pub feature_embedding: Option<Box<str>>,

    #[arg(
        long = "neighbour-k",
        default_value_t = 5,
        help = "How many embedding neighbours to add per cluster when `--feature-embedding` is set"
    )]
    pub neighbour_k: usize,

    #[arg(
        long = "fdr-alpha",
        default_value_t = 0.1,
        help = "FDR gate for calling a runner-up significant (must match annotate --fdr-alpha)"
    )]
    pub fdr_alpha: f32,

    #[arg(long, short = 'o', help = "Output prefix (default: `--from`)")]
    pub out: Option<Box<str>>,
}

#[derive(Debug, Clone)]
struct ClusterEvidence {
    id: String,
    coarse_label: String,
    best_label: String,
    best_q: Option<f32>,
    coarse_q: Option<f32>,
    label_support: Option<f32>,
    best_significant: bool,
    /// Second-lowest FDR-significant term `(label, q)`, when one exists beside the primary.
    second: Option<(String, f32)>,
    /// Marker genes/features that define the called (or closest) panel type.
    markers: Vec<String>,
    /// Extra genes pulled in via `--feature-embedding` neighbours of the markers.
    neighbour_genes: Vec<String>,
    incidence_words: Vec<String>,
}

impl ClusterEvidence {
    fn has_significant_call(&self) -> bool {
        self.best_significant && self.coarse_label != "unassigned"
    }
}

pub fn run_describe(args: &DescribeArgs) -> Result<()> {
    let out = args.out.as_deref().unwrap_or(args.from.as_ref());
    legume_numeric::matrix::common_io::mkdir_parent(out)?;

    let mut enriched = load_evidence(args.from.as_ref())?;
    attach_second_best(&mut enriched, args.from.as_ref(), args.fdr_alpha)?;
    let markers_by_type = load_markers_by_type(args.from.as_ref(), args.markers.as_deref())?;
    attach_markers(&mut enriched, &markers_by_type);
    if let Some(emb) = args.feature_embedding.as_deref() {
        attach_embedding_neighbours(&mut enriched, emb, args.neighbour_k)?;
    }
    attach_incidence_words(&mut enriched, args.text_prefix.as_deref())?;

    let allowed = allowed_entities(&enriched);
    let mut sentences = Vec::new();
    for c in &enriched {
        let draft = template_sentence(c);
        let sentence = citation_check(&draft, &allowed, c)?;
        sentences.push((c.id.clone(), sentence));
    }

    let evidence = json!({
        "from": args.from.as_ref(),
        "text_prefix": args.text_prefix.as_ref().map(|s| s.as_ref()),
        "markers": args.markers.as_ref().map(|s| s.as_ref()),
        "clusters": enriched.iter().map(|c| json!({
            "id": c.id,
            "coarse_label": c.coarse_label,
            "best_label": c.best_label,
            "best_q": c.best_q,
            "coarse_q": c.coarse_q,
            "label_support": c.label_support,
            "best_significant": c.best_significant,
            "second_label": c.second.as_ref().map(|(l, _)| l),
            "second_q": c.second.as_ref().map(|(_, q)| q),
            "markers": c.markers,
            "neighbour_genes": c.neighbour_genes,
            "incidence_words": c.incidence_words,
        })).collect::<Vec<_>>(),
        "allowed_entities": allowed.iter().cloned().collect::<Vec<_>>(),
    });

    let json_path = format!("{out}.describe.json");
    let mut jw = File::create(&json_path).with_context(|| format!("create {json_path}"))?;
    serde_json::to_writer_pretty(&mut jw, &evidence)?;
    writeln!(jw)?;
    info!("wrote {json_path}");

    let md_path = format!("{out}.describe.md");
    let mut mw = File::create(&md_path).with_context(|| format!("create {md_path}"))?;
    writeln!(mw, "# Cluster descriptions")?;
    writeln!(mw)?;
    writeln!(mw, "From `{from}`.", from = args.from.as_ref())?;
    writeln!(mw)?;
    for (id, s) in &sentences {
        writeln!(mw, "## {id}")?;
        writeln!(mw)?;
        writeln!(mw, "{s}")?;
        writeln!(mw)?;
    }
    info!("wrote {md_path} ({} cluster(s))", sentences.len());
    Ok(())
}

fn load_evidence(from: &str) -> Result<Vec<ClusterEvidence>> {
    let annot = format!("{from}.annot.parquet");
    let argmax = format!("{from}.argmax.tsv");
    let lineage_annot = format!("{from}.lineage_annot.annot.parquet");

    if Path::new(&annot).exists() {
        return load_from_annot_parquet(&annot);
    }
    if Path::new(&lineage_annot).exists() {
        return load_from_annot_parquet(&lineage_annot);
    }
    if Path::new(&argmax).exists() {
        return load_from_argmax_tsv(&argmax);
    }
    bail!(
        "no annotate artifacts under `{from}` \
         (expected `{from}.annot.parquet`, `{from}.lineage_annot.annot.parquet`, or `{from}.argmax.tsv`)"
    );
}

fn load_from_annot_parquet(path: &str) -> Result<Vec<ClusterEvidence>> {
    let fields = peek_parquet_field_names(path).with_context(|| format!("peek {path}"))?;
    let has = |name: &str| fields.iter().any(|f| f.as_ref() == name);
    anyhow::ensure!(
        has("coarse_label") && has("community"),
        "{path}: need coarse_label and community columns"
    );

    let string_cols: Vec<&str> = if has("best_label") {
        vec!["coarse_label", "best_label"]
    } else {
        vec!["coarse_label"]
    };
    let mut numeric_cols: Vec<&str> = vec!["community"];
    if has("best_q") {
        numeric_cols.push("best_q");
    }
    if has("coarse_q") {
        numeric_cols.push("coarse_q");
    }
    if has("label_support") {
        numeric_cols.push("label_support");
    }
    if has("best_significant") {
        numeric_cols.push("best_significant");
    }

    let (strings, nums) =
        legume_numeric::matrix::parquet::read_table_columns(path, &string_cols, &numeric_cols)
            .with_context(|| format!("read {path}"))?;

    let coarse = &strings[0];
    let best = if strings.len() > 1 {
        &strings[1]
    } else {
        coarse
    };
    let community: Vec<i32> = nums[0].iter().map(|x| *x as i32).collect();
    let best_q_col = numeric_cols.iter().position(|&c| c == "best_q");
    let coarse_q_col = numeric_cols.iter().position(|&c| c == "coarse_q");
    let support_col = numeric_cols.iter().position(|&c| c == "label_support");
    let best_sig_col = numeric_cols.iter().position(|&c| c == "best_significant");

    let mut by_comm: BTreeMap<i32, ClusterEvidence> = BTreeMap::new();
    for i in 0..community.len() {
        let id = community[i];
        by_comm.entry(id).or_insert_with(|| {
            let coarse_s = coarse[i].to_string();
            let best_s = best[i].to_string();
            ClusterEvidence {
                id: format!("K{id}"),
                coarse_label: coarse_s.clone(),
                best_label: best_s,
                best_q: best_q_col.map(|j| nums[j][i] as f32),
                coarse_q: coarse_q_col.map(|j| nums[j][i] as f32),
                label_support: support_col.map(|j| nums[j][i] as f32),
                best_significant: best_sig_col
                    .map(|j| nums[j][i] as i32 != 0)
                    .unwrap_or(coarse_s != "unassigned"),
                second: None,
                markers: Vec::new(),
                neighbour_genes: Vec::new(),
                incidence_words: Vec::new(),
            }
        });
    }
    Ok(by_comm.into_values().collect())
}

fn load_from_argmax_tsv(path: &str) -> Result<Vec<ClusterEvidence>> {
    let f = File::open(path).with_context(|| format!("open {path}"))?;
    let mut lines = BufReader::new(f).lines();
    let _header = lines.next().transpose()?;
    let mut labels: BTreeSet<String> = BTreeSet::new();
    for line in lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split('\t');
        let _cell = parts.next();
        if let Some(lab) = parts.next() {
            labels.insert(lab.to_string());
        }
    }
    Ok(labels
        .into_iter()
        .enumerate()
        .map(|(i, lab)| {
            let sig = lab != "unassigned";
            ClusterEvidence {
                id: format!("L{i}"),
                coarse_label: lab.clone(),
                best_label: lab,
                best_q: None,
                coarse_q: None,
                label_support: None,
                best_significant: sig,
                second: None,
                markers: Vec::new(),
                neighbour_genes: Vec::new(),
                incidence_words: Vec::new(),
            }
        })
        .collect())
}

/// Load panel markers keyed by cell-type name.
///
/// Preference: `{from}.marker_support.parquet` (bootstrap live genes),
/// then `{from}.marker_embedding.parquet`, then optional `--markers` TSV.
fn load_markers_by_type(
    from: &str,
    markers_tsv: Option<&str>,
) -> Result<BTreeMap<String, Vec<String>>> {
    let support = format!("{from}.marker_support.parquet");
    if Path::new(&support).exists() {
        return load_markers_from_support_parquet(&support);
    }
    let embed = format!("{from}.marker_embedding.parquet");
    if Path::new(&embed).exists() {
        return load_markers_from_embedding_parquet(&embed);
    }
    if let Some(path) = markers_tsv {
        return load_markers_from_tsv(path);
    }
    info!(
        "describe: no marker_support / marker_embedding under `{from}`; \
         pass --markers for a gene<TAB>celltype TSV"
    );
    Ok(BTreeMap::new())
}

fn load_markers_from_support_parquet(path: &str) -> Result<BTreeMap<String, Vec<String>>> {
    let fields = peek_parquet_field_names(path).with_context(|| format!("peek {path}"))?;
    let has = |name: &str| fields.iter().any(|f| f.as_ref() == name);
    anyhow::ensure!(
        has("gene") && has("cell_type"),
        "{path}: need gene and cell_type"
    );

    let string_cols = vec!["gene", "cell_type"];
    let mut numeric_cols: Vec<&str> = Vec::new();
    if has("live") {
        numeric_cols.push("live");
    }
    if has("idf_weight") {
        numeric_cols.push("idf_weight");
    }
    let (strings, nums) =
        legume_numeric::matrix::parquet::read_table_columns(path, &string_cols, &numeric_cols)
            .with_context(|| format!("read {path}"))?;
    let genes = &strings[0];
    let types = &strings[1];
    let live_col = numeric_cols.iter().position(|&c| c == "live");
    let weight_col = numeric_cols.iter().position(|&c| c == "idf_weight");

    // (weight, gene) per type — prefer live markers, higher IDF first.
    let mut scored: BTreeMap<String, Vec<(i32, f32, String)>> = BTreeMap::new();
    for i in 0..genes.len() {
        let live = live_col.map(|j| nums[j][i] as i32).unwrap_or(1);
        if live == 0 {
            continue;
        }
        let w = weight_col.map(|j| nums[j][i] as f32).unwrap_or(1.0);
        scored
            .entry(types[i].to_string())
            .or_default()
            .push((live, w, genes[i].to_string()));
    }
    Ok(finalize_marker_lists(scored))
}

fn load_markers_from_embedding_parquet(path: &str) -> Result<BTreeMap<String, Vec<String>>> {
    let fields = peek_parquet_field_names(path).with_context(|| format!("peek {path}"))?;
    let has = |name: &str| fields.iter().any(|f| f.as_ref() == name);
    anyhow::ensure!(has("gene") && has("type"), "{path}: need gene and type");

    let string_cols = ["gene", "type"];
    let numeric_cols: Vec<&str> = if has("weight") {
        vec!["weight"]
    } else {
        vec![]
    };
    let (strings, nums) =
        legume_numeric::matrix::parquet::read_table_columns(path, &string_cols, &numeric_cols)
            .with_context(|| format!("read {path}"))?;
    let genes = &strings[0];
    let types = &strings[1];
    let mut scored: BTreeMap<String, Vec<(i32, f32, String)>> = BTreeMap::new();
    for i in 0..genes.len() {
        let w = nums.first().map(|col| col[i] as f32).unwrap_or(1.0);
        scored
            .entry(types[i].to_string())
            .or_default()
            .push((1, w, genes[i].to_string()));
    }
    Ok(finalize_marker_lists(scored))
}

fn load_markers_from_tsv(path: &str) -> Result<BTreeMap<String, Vec<String>>> {
    let f = File::open(path).with_context(|| format!("open {path}"))?;
    let mut scored: BTreeMap<String, Vec<(i32, f32, String)>> = BTreeMap::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split('\t');
        let Some(gene) = parts.next() else {
            continue;
        };
        let Some(ct) = parts.next() else {
            continue;
        };
        if gene.eq_ignore_ascii_case("gene") || gene.eq_ignore_ascii_case("feature") {
            continue;
        }
        scored
            .entry(ct.to_string())
            .or_default()
            .push((1, 1.0, gene.to_string()));
    }
    Ok(finalize_marker_lists(scored))
}

fn finalize_marker_lists(
    scored: BTreeMap<String, Vec<(i32, f32, String)>>,
) -> BTreeMap<String, Vec<String>> {
    let mut out = BTreeMap::new();
    for (ct, mut rows) in scored {
        rows.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.2.cmp(&b.2))
        });
        let mut seen = BTreeSet::new();
        let mut genes = Vec::new();
        for (_, _, g) in rows {
            if seen.insert(g.clone()) {
                genes.push(g);
            }
            if genes.len() >= 12 {
                break;
            }
        }
        out.insert(ct, genes);
    }
    out
}

fn attach_markers(clusters: &mut [ClusterEvidence], by_type: &BTreeMap<String, Vec<String>>) {
    for c in clusters {
        let key = if c.has_significant_call() {
            c.coarse_label.as_str()
        } else if c.best_label != "unassigned" {
            c.best_label.as_str()
        } else {
            continue;
        };
        if let Some(genes) = by_type.get(key) {
            c.markers = genes.iter().take(8).cloned().collect();
        }
    }
}

/// Fill `second` from `{from}.cluster_term_q.parquet`.
///
/// Among terms with `q < fdr_alpha`, take the lowest-q term that is **not** the
/// primary call. Omitted when the runner-up is nonsignificant or the matrix is absent.
fn attach_second_best(clusters: &mut [ClusterEvidence], from: &str, fdr_alpha: f32) -> Result<()> {
    let path = format!("{from}.cluster_term_q.parquet");
    if !Path::new(&path).exists() {
        info!("describe: no {path}; skipping significant runner-up");
        return Ok(());
    }
    use legume_numeric::matrix::dmatrix_io::DMatrix;
    use legume_numeric::matrix::traits::IoOps;
    let loaded = DMatrix::<f32>::from_parquet_with_row_names(&path, Some(0))
        .with_context(|| format!("read {path}"))?;
    let row_of: BTreeMap<&str, usize> = loaded
        .rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.as_ref(), i))
        .collect();

    for c in clusters.iter_mut() {
        if !c.has_significant_call() {
            continue;
        }
        let Some(&ri) = row_of.get(c.id.as_str()) else {
            continue;
        };
        let primary = c.coarse_label.as_str();
        let mut ranked: Vec<(f32, &str)> = loaded
            .cols
            .iter()
            .enumerate()
            .filter_map(|(j, name)| {
                let q = loaded.mat[(ri, j)];
                if q.is_finite() && q < fdr_alpha {
                    Some((q, name.as_ref()))
                } else {
                    None
                }
            })
            .collect();
        ranked.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(b.1))
        });
        if let Some(&(q, lab)) = ranked.iter().find(|(_, lab)| *lab != primary) {
            c.second = Some((lab.to_string(), q));
        }
    }
    Ok(())
}

fn load_incidence_words(prefix: &str) -> Result<Vec<String>> {
    let path = format!("{prefix}.vocab.tsv");
    if !Path::new(&path).exists() {
        info!("describe: no {path}; skipping keyword incidence");
        return Ok(Vec::new());
    }
    // n_docs is only used for df_frac display in Vocabulary; 1 is fine for incidence listing.
    let vocab = Vocabulary::read_tsv(&path, 1)?;
    Ok(vocab.kept.iter().take(32).map(|w| w.to_string()).collect())
}

/// `feature → [(word, weight), …]` from `{prefix}.feature_word.edges.tsv`.
/// Keys are lowercased feature names. Only `* → word` rows are kept.
fn load_feature_word_edges(prefix: &str) -> Result<BTreeMap<String, Vec<(String, f32)>>> {
    let path = format!("{prefix}.feature_word.edges.tsv");
    if !Path::new(&path).exists() {
        return Ok(BTreeMap::new());
    }
    let f = File::open(&path).with_context(|| format!("open {path}"))?;
    let mut out: BTreeMap<String, Vec<(String, f32)>> = BTreeMap::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 5 {
            continue;
        }
        // lhs_type, lhs, rhs_type, rhs, weight
        if parts[2] != "word" {
            continue;
        }
        let gene = parts[1].to_lowercase();
        let word = parts[3].to_string();
        let w: f32 = parts[4].parse().unwrap_or(0.0);
        if w > 0.0 && !word.is_empty() {
            out.entry(gene).or_default().push((word, w));
        }
    }
    Ok(out)
}

fn gene_key(g: &str) -> String {
    g.to_lowercase()
}

/// Per-cluster keywords: sum feature→word weights over that cluster's markers
/// (and embedding neighbours), ranked by total weight.
///
/// Falls back to a global vocab head only when no `feature_word.edges.tsv` exists.
fn attach_incidence_words(
    clusters: &mut [ClusterEvidence],
    text_prefix: Option<&str>,
) -> Result<()> {
    let Some(prefix) = text_prefix else {
        return Ok(());
    };
    let edges = load_feature_word_edges(prefix)?;
    if edges.is_empty() {
        let fallback = load_incidence_words(prefix)?;
        if fallback.is_empty() {
            return Ok(());
        }
        info!(
            "describe: no feature_word edges under `{prefix}`; \
             using global vocab head (not per-cluster)"
        );
        for c in clusters.iter_mut() {
            c.incidence_words = fallback.iter().take(8).cloned().collect();
        }
        return Ok(());
    }

    for c in clusters.iter_mut() {
        let mut score: BTreeMap<String, f32> = BTreeMap::new();
        let genes = c.markers.iter().chain(c.neighbour_genes.iter());
        for g in genes {
            if let Some(words) = edges.get(&gene_key(g)) {
                for (word, w) in words {
                    *score.entry(word.clone()).or_default() += *w;
                }
            }
        }
        let mut ranked: Vec<(String, f32)> = score.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        c.incidence_words = ranked.into_iter().take(8).map(|(w, _)| w).collect();
    }
    Ok(())
}

/// Expand each cluster's gene set by cosine neighbours of the marker centroid
/// in a feature×H embedding (prefer `feature_coembedding` when available).
fn attach_embedding_neighbours(
    clusters: &mut [ClusterEvidence],
    emb_path: &str,
    k: usize,
) -> Result<()> {
    if k == 0 {
        return Ok(());
    }
    use legume_numeric::matrix::dmatrix_io::DMatrix;
    use legume_numeric::matrix::traits::IoOps;
    let loaded = DMatrix::<f32>::from_parquet_with_row_names(emb_path, Some(0))
        .with_context(|| format!("read {emb_path}"))?;
    let (n, h) = (loaded.mat.nrows(), loaded.mat.ncols());
    if n == 0 || h == 0 {
        return Ok(());
    }

    let mut index: BTreeMap<String, usize> = BTreeMap::new();
    for (i, name) in loaded.rows.iter().enumerate() {
        index.insert(gene_key(name), i);
    }

    // Precompute L2 norms for cosine.
    let norms: Vec<f32> = (0..n)
        .map(|i| {
            let mut s = 0.0f32;
            for j in 0..h {
                let v = loaded.mat[(i, j)];
                s += v * v;
            }
            s.sqrt().max(1e-12)
        })
        .collect();

    for c in clusters.iter_mut() {
        if c.markers.is_empty() {
            continue;
        }
        let mut centroid = vec![0.0f32; h];
        let mut n_hit = 0usize;
        let mut used = BTreeSet::new();
        for g in &c.markers {
            let Some(&ri) = index.get(&gene_key(g)) else {
                continue;
            };
            n_hit += 1;
            used.insert(ri);
            for (j, slot) in centroid.iter_mut().enumerate() {
                *slot += loaded.mat[(ri, j)];
            }
        }
        if n_hit == 0 {
            continue;
        }
        for v in &mut centroid {
            *v /= n_hit as f32;
        }
        let cnorm = centroid
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);

        let mut scored: Vec<(f32, usize)> = Vec::with_capacity(n);
        for (i, &gnorm) in norms.iter().enumerate() {
            if used.contains(&i) {
                continue;
            }
            let mut dot = 0.0f32;
            for (j, &cv) in centroid.iter().enumerate() {
                dot += cv * loaded.mat[(i, j)];
            }
            scored.push((dot / (cnorm * gnorm), i));
        }
        if scored.len() > k {
            scored.select_nth_unstable_by(k, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(k);
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        c.neighbour_genes = scored
            .into_iter()
            .map(|(_, i)| loaded.rows[i].to_string())
            .collect();
    }
    info!("describe: expanded markers with up to {k} neighbours from {emb_path}");
    Ok(())
}

fn cluster_entities(c: &ClusterEvidence) -> BTreeSet<String> {
    let mut s = BTreeSet::new();
    if c.coarse_label != "unassigned" {
        s.insert(c.coarse_label.clone());
    }
    if c.best_label != "unassigned" {
        s.insert(c.best_label.clone());
    }
    if let Some((lab, _)) = &c.second {
        s.insert(lab.clone());
    }
    s.extend(c.incidence_words.iter().cloned());
    s.extend(c.markers.iter().cloned());
    // Neighbours are evidence for keyword fishing, not citation-checked panel types.
    s
}

fn allowed_entities(clusters: &[ClusterEvidence]) -> BTreeSet<String> {
    let mut s = BTreeSet::new();
    for c in clusters {
        s.extend(cluster_entities(c));
        s.extend(c.neighbour_genes.iter().cloned());
    }
    s.insert("unassigned".into());
    s
}

fn significance_clause(c: &ClusterEvidence, omit_best_q: bool) -> String {
    let mut parts = Vec::new();
    if !omit_best_q {
        if let Some(q) = c.best_q.filter(|q| q.is_finite()) {
            parts.push(format!("best q={q:.3}"));
        }
    }
    if let Some(q) = c.coarse_q.filter(|q| q.is_finite()) {
        if c.best_q.map(|b| (b - q).abs() > 1e-6).unwrap_or(true) {
            parts.push(format!("coarse q={q:.3}"));
        }
    }
    if let Some(s) = c.label_support.filter(|s| s.is_finite()) {
        parts.push(format!("label support={s:.2}"));
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(" ({})", parts.join("; "))
    }
}

fn marker_sentence(c: &ClusterEvidence, omit_best_q: bool) -> String {
    let sig = significance_clause(c, omit_best_q);
    if c.markers.is_empty() {
        if sig.is_empty() {
            return String::new();
        }
        return if c.has_significant_call() {
            format!(" Significance:{sig}.")
        } else {
            format!(" Closest-type significance:{sig}.")
        };
    }
    let genes = c
        .markers
        .iter()
        .take(5)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if c.has_significant_call() {
        format!(" The call is supported by markers {genes}{sig}.")
    } else {
        format!(" Closest-type markers include {genes}{sig}.")
    }
}

fn runner_up_clause(c: &ClusterEvidence) -> String {
    match &c.second {
        Some((lab, q)) if c.has_significant_call() => {
            format!(" A second significant contender is {lab} (q={q:.3}).")
        }
        _ => String::new(),
    }
}

fn template_sentence(c: &ClusterEvidence) -> String {
    let words = if c.incidence_words.is_empty() {
        String::new()
    } else {
        format!(
            " Keywords from marker text: {}.",
            c.incidence_words
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let runner = runner_up_clause(c);
    if c.has_significant_call() {
        let markers = marker_sentence(c, false);
        format!(
            "{} is annotated as {}.{}{}{}",
            c.id, c.coarse_label, markers, runner, words
        )
    } else if c.best_label != "unassigned" {
        let q = c
            .best_q
            .map(|q| format!(" (best q={q:.3})"))
            .unwrap_or_default();
        // best q already appears in the lead clause — omit it from the marker sentence.
        let markers = marker_sentence(c, c.best_q.is_some());
        format!(
            "{} has no significant call; closest panel type is {}{}.{}{}",
            c.id, c.best_label, q, markers, words
        )
    } else {
        let markers = marker_sentence(c, false);
        format!("{} is unassigned.{}{}", c.id, markers, words)
    }
}

fn citation_check(draft: &str, allowed: &BTreeSet<String>, c: &ClusterEvidence) -> Result<String> {
    let lower = draft.to_lowercase();
    let cluster_allowed = cluster_entities(c);
    if !cluster_allowed.is_empty() {
        let mentions = cluster_allowed
            .iter()
            .any(|e| lower.contains(&e.to_lowercase()));
        if !mentions {
            bail!(
                "citation check failed for {}: sentence names none of {:?}",
                c.id,
                cluster_allowed
            );
        }
    }
    // Reject other panel types from the run that are not this cluster's evidence.
    // Gene symbols can collide; only enforce for entities that are not markers/words.
    let panel_types: BTreeSet<String> = allowed
        .iter()
        .filter(|e| {
            e.as_str() != "unassigned"
                && !c.markers.iter().any(|g| g == *e)
                && !c.incidence_words.iter().any(|w| w == *e)
                && !c.neighbour_genes.iter().any(|g| g == *e)
        })
        .cloned()
        .collect();
    for ent in &panel_types {
        if cluster_allowed.contains(ent) {
            continue;
        }
        if lower.contains(&ent.to_lowercase())
            && !cluster_allowed.iter().any(|a| {
                a.to_lowercase().contains(&ent.to_lowercase())
                    || ent.to_lowercase().contains(&a.to_lowercase())
            })
            && ent.len() >= 3
        {
            bail!(
                "citation check failed for {}: sentence mentions `{}` outside evidence",
                c.id,
                ent
            );
        }
    }
    if draft.contains('{') || draft.contains('}') {
        bail!("citation check failed for {}: braces in sentence", c.id);
    }
    Ok(draft.to_string())
}

#[cfg(test)]
mod tests;
