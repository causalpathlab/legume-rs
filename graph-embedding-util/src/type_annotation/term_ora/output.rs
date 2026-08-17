//! Everything the run leaves behind: the `{out_prefix}.*` artifacts, the console reports
//! that name what went wrong, and step 7's ontology hand-off.
//!
//! Writers and reports live together because they answer the same question from two ends —
//! the parquet/TSV carries the per-cell and per-cluster numbers, while the `warn!`s beside
//! them say which of those numbers should not be believed (a panel no better than random
//! genes, a type assigned by noise, a permutation null with no spread).

use super::ora::OraResult;
use super::{label_of, TermOraConfig};
use crate::type_annotation::marker_bootstrap::{BootstrapResult, CoarseConsensus};
use crate::type_annotation::UNASSIGNED;
use anyhow::{Context, Result};
use log::{info, warn};
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::parquet::{write_named_table, Column};
use matrix_util::traits::IoOps;
use std::io::Write;

#[allow(clippy::too_many_arguments)]
pub(super) fn write_annot_parquet(
    out_prefix: &str,
    cell_names: &[Box<str>],
    community: &[usize],
    sizes: &[usize],
    coarse_label: &[Box<str>],
    assign: &[usize],
    dist: &[f32],
    type_names: &[Box<str>],
    ora: &OraResult,
    cluster_label: &[usize],
    boot: Option<&BootstrapResult>,
    consensus: Option<&CoarseConsensus>,
    sup_null: Option<&crate::type_annotation::support_null::SupportNull>,
) -> Result<()> {
    let n = cell_names.len();
    let c = type_names.len();
    let comm_i32: Vec<i32> = community.iter().map(|&k| k as i32).collect();
    // How many cells the call was pooled over — its cluster's size. The test's power comes
    // entirely from this number, and a call resting on a handful of cells is a different animal
    // from one resting on hundreds, so it goes out in the parquet rather than being implied.
    let cluster_size: Vec<i32> = community.iter().map(|&k| sizes[k] as i32).collect();
    let fine_label: Vec<Box<str>> = assign.iter().map(|&t| label_of(t, type_names)).collect();
    let is_outlier: Vec<i32> = assign.iter().map(|&t| (t == UNASSIGNED) as i32).collect();
    // Per-cell coarse stats = the cluster's call entry, broadcast to its members.
    let stat_of = |m: &[f32], i: usize| -> f32 {
        let k = community[i];
        match cluster_label[k] {
            UNASSIGNED => f32::NAN,
            t => m[k * c + t],
        }
    };
    let coarse_p: Vec<f32> = (0..n).map(|i| stat_of(&ora.p_perm, i)).collect();
    let coarse_q: Vec<f32> = (0..n).map(|i| stat_of(&ora.q, i)).collect();

    let annot_path = format!("{out_prefix}.annot.parquet");
    let mut cols = vec![
        (Box::from("community"), Column::I32(&comm_i32)),
        (Box::from("cluster_size"), Column::I32(&cluster_size)),
        (Box::from("coarse_label"), Column::Str(coarse_label)),
        (Box::from("fine_label"), Column::Str(&fine_label)),
        (Box::from("fine_distance"), Column::F32(dist)),
        (Box::from("is_outlier"), Column::I32(&is_outlier)),
    ];
    // **`coarse_p`/`coarse_q` are only honest without the bootstrap**, and are withheld with it.
    //
    // They are one partition's word: the p/q that this single (irreproducible) Leiden run gave
    // to the term *it* picked for the cell's cluster. Under the bootstrap `coarse_label` is the
    // consensus over resampled panels and re-partitionings instead — so the two disagree about
    // which term they even describe, and they disagree about how sure to be. Measured on cord
    // blood: 6,891 cells whose consensus label was `NK` carried q between 1e-3 and 6e-3 — flat
    // certainty — next to a `label_support` of 0.50-0.60, a coin flip. Shipping a p-value that
    // confident beside a label that unstable is worse than shipping no p-value at all, and the
    // p-value is the one that is lying.
    if boot.is_none() {
        cols.extend([
            (Box::from("coarse_p"), Column::F32(&coarse_p)),
            (Box::from("coarse_q"), Column::F32(&coarse_q)),
        ]);
    }
    // The bootstrap's per-cell numbers, which are what replace them. These vary cell by cell —
    // the whole point of resampling — where a cluster's p/q is identical for every one of its
    // members. `label_support` is the headline: the fraction of replicates (panel resampled, and
    // the partition re-derived) that agreed on this cell's shipped label.
    // The **mixed annotation**. `coarse_label` is forced to pick one type or give up; this is
    // what the resampling actually said, and it is defined for every cell — including the ones
    // `coarse_label` abstains on, which is exactly where it earns its keep. `HSPC/LMPP` is a real
    // answer; `unassigned` is a refusal to give one.
    //
    // **The set is rendered in canonical (type-index) order, not in support order.** A
    // set-valued label is a category, and a category has to have one spelling: sorting by support
    // would render the same 3-way call as `Erythroid/Granulo-Mono/HSPC` for one cell and
    // `Granulo-Mono/Erythroid/HSPC` for the next, so grouping by it would split one group into
    // `k!` of them. Which member is the most probable is already carried by `coarse_label` and
    // `label_support`; this column's job is to name the *set*.
    let set_str: Vec<Box<str>>;
    let ranked_str: Vec<Box<str>>;
    let set_size: Vec<i32>;
    if let (Some(b), Some(con)) = (boot, consensus) {
        // One spelling of a label set: `unassigned` when empty (too wide to mean anything —
        // see `CoarseConsensus::label_set`), else `label_of` joined by "/". `sort` canonicalises
        // by type index (a set is a *category* with one spelling); leaving it unsorted keeps the
        // credible-set support order (leading first) that `label_ranked` needs.
        let join_set = |set: &[usize], sort: bool| -> Box<str> {
            if set.is_empty() {
                return Box::from(enrichment::UNASSIGNED_LABEL);
            }
            let mut ix = set.to_vec();
            if sort {
                ix.sort_unstable(); // the `unassigned` column is `c`, so it sorts last
            }
            ix.iter()
                .map(|&t| label_of(t, type_names))
                .collect::<Vec<_>>()
                .join("/")
                .into_boxed_str()
        };
        // `label_set` names the *category* (canonicalised); `label_ranked` is its support-ordered
        // twin (largest share first) — the plot reads the leading fate + runner-up from it.
        set_str = con.label_set.iter().map(|s| join_set(s, true)).collect();
        ranked_str = con.label_set.iter().map(|s| join_set(s, false)).collect();
        set_size = con.label_set.iter().map(|s| s.len() as i32).collect();
        cols.extend([
            (Box::from("label_set"), Column::Str(&set_str)),
            (Box::from("label_ranked"), Column::Str(&ranked_str)),
            (Box::from("label_set_size"), Column::I32(&set_size)),
            (
                Box::from("label_set_support"),
                Column::F32(&con.set_support),
            ),
            (Box::from("label_support"), Column::F32(&con.support)),
            (Box::from("label_entropy"), Column::F32(&con.entropy)),
            (Box::from("fine_support"), Column::F32(&b.support)),
            (Box::from("fine_entropy"), Column::F32(&b.entropy)),
        ]);
    }
    // The support, calibrated: `support_q` is a Benjamini–Hochberg FDR across the cells, so a
    // cutoff on it means the same thing whatever the number of types — unlike `label_support`,
    // whose natural scale is `1/C`.
    if let Some(sn) = sup_null {
        cols.extend([
            (Box::from("support_p"), Column::F32(&sn.p)),
            (Box::from("support_q"), Column::F32(&sn.q)),
            (Box::from("null_support"), Column::F32(&sn.null_support)),
        ]);
    }
    write_named_table(&annot_path, "cell", cell_names, &cols)
        .with_context(|| format!("writing {annot_path}"))?;
    info!("wrote {annot_path}");
    Ok(())
}

///////////////////////////////////////
// marker-bootstrap outputs + report //
///////////////////////////////////////

/// The per-cell bootstrap distribution, the marker-deviation table, and the per-type
/// diagnostics.
///
/// `marker_deviation` is the deliverable that states the first error source — *does the
/// embedding actually place this listed gene anywhere near the type it is listed under?* —
/// as an output rather than absorbing it silently into a centroid.
pub(super) fn write_bootstrap_outputs(
    out_prefix: &str,
    cell_names: &[Box<str>],
    gene_names: &[Box<str>],
    type_names: &[Box<str>],
    type_markers: &[Vec<(u32, f32)>],
    post: &BootstrapResult,
    consensus: &CoarseConsensus,
) -> Result<()> {
    // The distribution over the SHIPPED label — what fraction of the replicates put this cell
    // in each type, and in `unassigned`.
    let mut col_names: Vec<Box<str>> = type_names.to_vec();
    col_names.push(Box::from(enrichment::UNASSIGNED_LABEL));
    let path = format!("{out_prefix}.label_stability.parquet");
    DMatrix::<f32>::from_row_iterator(cell_names.len(), post.c + 1, consensus.post.iter().copied())
        .to_parquet_with_names(&path, (Some(cell_names), Some("cell")), Some(&col_names))
        .with_context(|| format!("writing {path}"))?;
    info!("wrote {path}");

    // Long (gene, type) marker table.
    let (mut genes, mut types, mut weights, mut dev, mut live) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (t, markers) in type_markers.iter().enumerate() {
        for (j, &(gi, w)) in markers.iter().enumerate() {
            genes.push(gene_names[gi as usize].clone());
            types.push(type_names[t].clone());
            weights.push(w);
            dev.push(post.marker_dev[t][j]);
            live.push(i32::from(post.marker_live[t][j]));
        }
    }
    let path = format!("{out_prefix}.marker_support.parquet");
    write_named_table(
        &path,
        "gene",
        &genes,
        &[
            (Box::from("cell_type"), Column::Str(&types)),
            (Box::from("idf_weight"), Column::F32(&weights)),
            (Box::from("deviation"), Column::F32(&dev)),
            (Box::from("live"), Column::I32(&live)),
        ],
    )
    .with_context(|| format!("writing {path}"))?;
    info!("wrote {path}");

    let path = format!("{out_prefix}.type_qc.tsv");
    let mut f = std::fs::File::create(&path).with_context(|| format!("creating {path}"))?;
    // `n_draws` is the number of replicates that actually ran — every support in this run is a
    // fraction of it, and a Ctrl+C makes it smaller than `--n-boot`.
    writeln!(
        f,
        "cell_type\tn_draws\tn_markers\tn_live\tcentroid_jitter\tdecision_gap\tnoise_ratio\tmean_support\toccupancy"
    )?;
    for (t, qc) in post.type_qc.iter().enumerate() {
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.3}\t{:.4}\t{:.4}",
            type_names[t],
            post.n_draws,
            type_markers[t].len(),
            qc.n_live,
            qc.centroid_jitter,
            qc.decision_gap,
            noise_ratio(qc),
            qc.mean_support,
            qc.occupancy
        )?;
    }
    info!("wrote {path}");
    Ok(())
}

/// The panel null's per-type verdict: does the type's own gene list place its prototype better
/// than the same number of random marker genes would?
pub(super) fn write_panel_null(
    out_prefix: &str,
    type_names: &[Box<str>],
    pn: &crate::type_annotation::panel_null::PanelNull,
) -> Result<()> {
    let path = format!("{out_prefix}.panel_null.tsv");
    let mut f = std::fs::File::create(&path).with_context(|| format!("creating {path}"))?;
    writeln!(
        f,
        "cell_type\tn_live\toccupancy\tnull_occupancy\tcost\tnull_cost\tcost_ratio\tp"
    )?;
    for (t, name) in type_names.iter().enumerate() {
        writeln!(
            f,
            "{name}\t{}\t{:.4}\t{:.4}\t{:.1}\t{:.1}\t{:.3}\t{:.4}",
            pn.n_live[t],
            pn.occupancy[t],
            pn.null_occupancy[t],
            pn.cost[t],
            pn.null_cost[t],
            pn.null_cost[t] / pn.cost[t].max(1e-9),
            pn.p[t]
        )?;
    }
    info!("wrote {path}");
    Ok(())
}

/// Name the types whose gene list is doing no better than random genes would. These are the types
/// the point-estimate path fills anyway, confidently, and that the bootstrap will *also* call
/// confidently — because every resample of a wrong panel is wrong the same way.
pub(super) fn report_panel_null(
    pn: &crate::type_annotation::panel_null::PanelNull,
    type_names: &[Box<str>],
) {
    let mut dud: Vec<(&str, f32, f32)> = type_names
        .iter()
        .enumerate()
        .filter(|&(t, _)| pn.n_live[t] > 0 && pn.p[t] > 0.05)
        .map(|(t, name)| (name.as_ref(), pn.p[t], pn.occupancy[t]))
        .collect();
    info!(
        "marker-panel null ({} draws/type): {}/{} types place their prototype better than random \
         genes of the same number (p < 0.05)",
        pn.n_perm,
        type_names.len() - dud.len(),
        type_names.len()
    );
    if dud.is_empty() {
        return;
    }
    dud.sort_by(|a, b| b.1.total_cmp(&a.1));
    let preview: Vec<String> = dud
        .iter()
        .take(10)
        .map(|&(name, p, occ)| format!("{name} (p={p:.2}, holds {:.1}% of cells)", occ * 100.0))
        .collect();
    warn!(
        "{} type(s) are NOT identified by their own markers — random genes of the same number \
         place their prototype just as well, yet they still hold cells: {}. This is BIAS, and no \
         amount of bootstrapping will find it: every resample of a wrong panel is wrong the same \
         way, so these calls come back *stable*. See {{out}}.panel_null.tsv.",
        dud.len(),
        preview.join(", ")
    );
}

/// How far the centroid moves under resampling, against the margin the assignment is actually
/// decided by. **Above 1 the type's cells are being assigned by noise**: the panel does not
/// place the type to within the precision the decision needs.
fn noise_ratio(qc: &crate::type_annotation::marker_bootstrap::TypeQc) -> f32 {
    if qc.decision_gap > 0.0 {
        qc.centroid_jitter / qc.decision_gap
    } else {
        f32::NAN
    }
}

/// Say plainly how much of the labelling was resting on arbitrary choices: how far the
/// clustering wandered across replicates, and how many cells could not hold a label through it.
pub(super) fn report_consensus(con: &CoarseConsensus, n: usize) {
    let called = con.label.iter().filter(|&&t| t != UNASSIGNED).count();
    let mean_support = con.support.iter().sum::<f32>() / n as f32;
    let (lo, hi) = con
        .n_comm
        .iter()
        .fold((usize::MAX, 0), |(l, h), &k| (l.min(k), h.max(k)));
    info!(
        "stability bootstrap ({} replicates, panel + clustering resampled): {called}/{n} cells \
         held a label (mean support {mean_support:.2}); the clustering itself ranged over \
         {lo}–{hi} communities across replicates",
        con.n_comm.len()
    );
    if hi > 2 * lo.max(1) {
        warn!(
            "the clustering is unstable — replicates ranged from {lo} to {hi} communities on the \
             SAME data. Any single partition's labelling is one draw from that, which is why the \
             shipped label is now the consensus rather than one run's word for it."
        );
    }
}

/// Name the types whose panel cannot place them to the precision the assignment needs. These
/// are exactly the types the point-estimate path hands a confident share of cells anyway.
pub(super) fn report_bootstrap(post: &BootstrapResult, type_names: &[Box<str>]) {
    let mut weak: Vec<(&str, f32, f32)> = post
        .type_qc
        .iter()
        .zip(type_names)
        .filter(|(qc, _)| noise_ratio(qc) > 1.0)
        .map(|(qc, name)| (name.as_ref(), noise_ratio(qc), qc.occupancy))
        .collect();
    if weak.is_empty() {
        return;
    }
    weak.sort_by(|a, b| b.1.total_cmp(&a.1));
    let preview: Vec<String> = weak
        .iter()
        .take(10)
        .map(|&(name, r, occ)| format!("{name} ({r:.1}×, {occ:.1}% of cells)", occ = occ * 100.0))
        .collect();
    warn!(
        "{} type(s) move further under marker resampling than the margin their assignment is \
         decided by — their cells are being called by noise: {}. This is an EMBEDDING problem, \
         not a statistics one: re-run `senna gem --must-train-features <panel>` so the marker \
         genes are trained rather than post-hoc projected. See {{out}}.type_qc.tsv.",
        weak.len(),
        preview.join(", ")
    );
}

pub(super) fn write_cluster_term_matrices(
    out_prefix: &str,
    comm_names: &[Box<str>],
    type_names: &[Box<str>],
    ora: &OraResult,
) -> Result<()> {
    let n_comm = comm_names.len();
    let c = type_names.len();
    let to_mat = |flat: &[f32]| DMatrix::<f32>::from_row_iterator(n_comm, c, flat.iter().copied());
    for (suffix, flat) in [
        ("cluster_term_p", &ora.p_perm),
        ("cluster_term_q", &ora.q),
        ("cluster_term_softq", &ora.q_soft),
    ] {
        let path = format!("{out_prefix}.{suffix}.parquet");
        to_mat(flat)
            .to_parquet_with_names(&path, (Some(comm_names), Some("cluster")), Some(type_names))
            .with_context(|| format!("writing {path}"))?;
    }
    info!("wrote {out_prefix}.cluster_term_{{p,q,Q}}.parquet ({n_comm} clusters × {c} terms)");
    Ok(())
}

pub(super) fn write_calibration(
    out_prefix: &str,
    ora: &OraResult,
    n_assigned: usize,
    n_outliers: usize,
) -> Result<()> {
    let Some(cal) = ora.cal.as_ref() else {
        return Ok(()); // a `Want::CallOnly` run has nothing to report
    };
    let path = format!("{out_prefix}.null_calibration.tsv");
    let mut f = std::fs::File::create(&path).with_context(|| format!("creating {path}"))?;
    writeln!(f, "metric\tvalue")?;
    writeln!(f, "n_perm\t{}", cal.n_perm)?;
    writeln!(f, "n_assigned\t{n_assigned}")?;
    writeln!(f, "n_outliers_pruned\t{n_outliers}")?;
    writeln!(
        f,
        "median_logratio_perm_over_analytic\t{:.4}",
        cal.median_logratio
    )?;
    writeln!(
        f,
        "frac_analytic_anticonservative\t{:.4}",
        cal.frac_analytic_anticons
    )?;
    writeln!(f, "lambda_perm\t{:.4}", cal.lambda_perm)?;
    writeln!(f, "ks_perm_uniform\t{:.4}", cal.ks_perm)?;
    writeln!(f, "degenerate_frac\t{:.4}", cal.degenerate_frac)?;
    info!("wrote {path}");

    // Console summary + warnings.
    eprintln!("\nNull calibration (permutation B={})", cal.n_perm);
    eprintln!(
        "  analytic vs permutation:  median log10(p_perm/p_analytic)={:.3}  anticonservative-frac={:.3}",
        cal.median_logratio, cal.frac_analytic_anticons
    );
    eprintln!(
        "  permutation machinery:    lambda_perm={:.3}  ks_uniform={:.3}  degenerate-frac={:.3}",
        cal.lambda_perm, cal.ks_perm, cal.degenerate_frac
    );
    if cal.median_logratio > 0.3 || cal.frac_analytic_anticons > 0.2 {
        log::warn!(
            "analytic hypergeometric looks anticonservative (median log-ratio {:.2}); \
             the reported p/q use the permutation null",
            cal.median_logratio
        );
    }
    // λ straying from 1 has two very different causes, and telling the user to raise --num-perm
    // is right for exactly one of them. When a term's pooled null has no spread — too few cells
    // are assigned to it for any relabeling to move its count — its statistic is a constant, and
    // no number of permutations will ever give that constant a distribution. `degenerate_frac`
    // is precisely the share of terms in that state, so let it pick the message.
    if cal.lambda_perm.is_finite() && !(0.7..=1.4).contains(&cal.lambda_perm) {
        if cal.degenerate_frac > 0.1 {
            log::warn!(
                "permutation null lambda_perm={:.2} strays from 1, but {:.0}% of terms have a \
                 null with no spread — too few cells are assigned to them to test at all. More \
                 permutations cannot fix this; a marker panel the embedding actually trained on \
                 can (`senna gem --must-train-features`).",
                cal.lambda_perm,
                cal.degenerate_frac * 100.0
            );
        } else {
            log::warn!(
                "permutation null lambda_perm={:.2} strays from 1 — raise --num-perm",
                cal.lambda_perm
            );
        }
    }
    eprintln!();
    Ok(())
}

/// Print each cluster's call, largest first. A sane partition has tens of clusters; a runaway
/// one (`--resolution 8` has produced 1,713 on 15k cells) is truncated rather than allowed to
/// bury the log — and the truncation is stated, not silent.
const MAX_CLUSTERS_LISTED: usize = 64;

pub(super) fn log_cluster_calls(cluster_label: &[usize], type_names: &[Box<str>], sizes: &[usize]) {
    let n_comm = sizes.len();
    info!("cluster calls ({n_comm} clusters):");
    let mut order: Vec<usize> = (0..n_comm).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(sizes[k]));
    for &k in order.iter().take(MAX_CLUSTERS_LISTED) {
        let name = label_of(cluster_label[k], type_names);
        info!("  K{k:<3} {:6} cells  {name}", sizes[k]);
    }
    if let Some(rest) = n_comm.checked_sub(MAX_CLUSTERS_LISTED).filter(|&r| r > 0) {
        info!("  … and {rest} smaller cluster(s)");
    }
}

/////////////////
// 7. ontology //
/////////////////

pub(super) fn run_ontology(
    out_prefix: &str,
    obo: &str,
    label_cl: &str,
    comm_names: &[Box<str>],
    type_names: &[Box<str>],
    ora: &OraResult,
    cfg: &TermOraConfig,
) -> Result<()> {
    let n_comm = comm_names.len();
    let c = type_names.len();
    // cluster × term permutation p as the ontology leaf evidence; Q as node mass.
    let p_mat = DMatrix::<f32>::from_row_iterator(n_comm, c, ora.p_perm.iter().copied());
    let q_mat = DMatrix::<f32>::from_row_iterator(n_comm, c, ora.q_soft.iter().copied());
    crate::type_annotation::ontology_obo::annotate_ontology_from_obo(
        out_prefix,
        label_cl,
        obo,
        cfg.ontology_fdr_q,
        cfg.ontology_by,
        enrichment::OntologyScore::Pvalue(&p_mat),
        Some(&q_mat),
        comm_names,
        type_names,
    )?;
    Ok(())
}
