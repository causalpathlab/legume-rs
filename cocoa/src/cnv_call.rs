//! CNV calling on cocoa's confounder-adjusted pseudobulk parameters.
//!
//! Pipeline:
//! 1. Build `[G × (I·K)]` log-ratio signal from `parameters[k].exposure`
//!    (per-individual log-mean after confounder adjustment).
//!    Within each topic, auto-detect "normal" reference samples (lowest
//!    genome-wide deviation from the per-gene median) and normalize each
//!    sample against the reference per-gene mean.
//! 2. Hand off to [`cnv::per_sample::call_per_sample_cnv`] for the per-topic
//!    HMM-EM fit; optionally iterate the reference set 2-3 times by
//!    re-clustering samples on cn_score and picking the lower-burden cluster.
//! 3. Emit:
//!    * `{out}.cnv.states.parquet` — `[G × N]` Viterbi state per sample
//!    * `{out}.cnv.cn_score.parquet` — `[G × N]` continuous CN score in [−1, 1]
//!    * `{out}.cnv.normal_samples.tsv.gz` — auto-detected reference samples per topic
//!    * `{out}.cnv.segments.bed.gz` — per-sample non-neutral segments, BED6
//! 4. Annotate the DEG perm output with `cnv_concordance_r` / `cnv_concordance_p`:
//!    per-gene Pearson correlation of the gene's signal vs. its own cn_score.

use crate::common::*;
use crate::stat::{z_to_pvalue, CocoaGammaOut};

use clap::Args;
use cnv::genome_order::GenePosition;
use cnv::per_sample::{
    call_per_sample_cnv, cluster_reference_from_cn_score, detect_normal_samples, modal_state_at,
    topic_local_to_flat, PerSampleCnv, PerSampleCnvConfig,
};
use matrix_param::traits::Inference;
use matrix_util::common_io::write_lines;
use matrix_util::traits::IoOps;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

#[derive(Args, Debug, Clone)]
pub struct CnvArgs {
    #[arg(
        long,
        help = "GFF/GTF annotation for CNV detection.",
        long_help = "GFF/GTF file with gene coordinates. When provided, runs the\n\
                     per-sample HMM CNV model on the cocoa-adjusted (indv × topic)\n\
                     log-ratio matrix."
    )]
    pub gff: Option<Box<str>>,

    #[arg(
        long,
        help = "CNV ground-truth TSV (alternative to --gff).",
        long_help = "Alternative to --gff: reads gene coordinates from a\n\
                     `.cnv_ground_truth.tsv.gz` file (gene_idx, chr, pos, state).\n\
                     Useful for simulation studies."
    )]
    pub cnv_ground_truth: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of CN states (3 = del/neutral/gain; 5/6 = inferCNV i6-style)."
    )]
    pub cnv_states: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Iterative reference refinement passes (1 = no refinement).",
        long_help = "After the first HMM call, cluster samples within each topic on\n\
                     their per-gene cn_score (kmeans K=2). The lower-burden cluster\n\
                     becomes the new reference; signal is rebuilt and HMM re-run."
    )]
    pub cnv_iter_ref: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "If ≥ 3, BIC-select K ∈ [3..max] via kmeans on the marginal signal."
    )]
    pub cnv_gmm_k_max: usize,
}

/// Reference fraction used as initial-pass normalisation (lowest-dispersion samples).
const CNV_NORMAL_FRAC: f32 = 0.33;

impl CnvArgs {
    pub fn enabled(&self) -> bool {
        self.gff.is_some() || self.cnv_ground_truth.is_some()
    }

    fn per_sample_config(&self) -> PerSampleCnvConfig {
        PerSampleCnvConfig {
            n_states: self.cnv_states,
            gmm_k_max: self.cnv_gmm_k_max,
            ..PerSampleCnvConfig::default()
        }
    }
}

////////////////////
// Gene positions //
////////////////////

pub fn load_gene_positions(
    args: &CnvArgs,
    gene_names: &[Box<str>],
) -> anyhow::Result<Option<Vec<GenePosition>>> {
    if let Some(path) = &args.cnv_ground_truth {
        info!("CNV: loading gene positions from {}", path);
        return Ok(Some(cnv::genome_order::read_gene_positions_from_tsv(path)?));
    }
    if let Some(path) = &args.gff {
        info!("CNV: loading gene annotations from {}", path);
        let gene_tss = genomic_data::coordinates::load_gene_tss(path, gene_names)?;
        let positions: Vec<GenePosition> = gene_tss
            .iter()
            .enumerate()
            .filter_map(|(idx, tss)| {
                tss.as_ref().map(|t| GenePosition {
                    gene_idx: idx,
                    chromosome: t.chr.clone(),
                    position: t.tss as u64,
                })
            })
            .collect();
        info!(
            "CNV: matched {} / {} genes to GFF entries",
            positions.len(),
            gene_names.len()
        );
        return Ok(Some(positions));
    }
    Ok(None)
}

/////////////////////////
// Signal construction //
/////////////////////////

/// Build per-topic normalised signal `[G × (I·K)]` given per-topic reference
/// indices (topic-local `0..n_indv`). Cocoa-specific because it reads from
/// the `CocoaGammaOut.exposure` posterior log-mean.
fn build_cnv_signal_with_refs(
    parameters: &[CocoaGammaOut],
    indv_names: &[Box<str>],
    topic_names: &[Box<str>],
    references_per_topic: &[Vec<usize>],
) -> (DMatrix<f32>, Vec<Box<str>>) {
    let n_topics = parameters.len();
    assert_eq!(references_per_topic.len(), n_topics);
    let log_tau0 = parameters[0].exposure.posterior_log_mean();
    let n_genes = log_tau0.nrows();
    let n_indv = log_tau0.ncols();
    let total_cols = n_indv * n_topics;

    let mut signal = DMatrix::<f32>::zeros(n_genes, total_cols);
    let mut sample_names: Vec<Box<str>> = Vec::with_capacity(total_cols);

    for k in 0..n_topics {
        let log_tau = parameters[k].exposure.posterior_log_mean();
        let refs = &references_per_topic[k];
        let nn = (refs.len() as f32).max(1.0);
        let mut reference_g = DVector::<f32>::zeros(n_genes);
        for &i in refs {
            reference_g += log_tau.column(i);
        }
        reference_g.scale_mut(1.0 / nn);

        let topic = topic_names.get(k).map(|s| s.as_ref()).unwrap_or("topic");
        for (i, indv) in indv_names.iter().enumerate().take(n_indv) {
            let col = k * n_indv + i;
            let mut dst = signal.column_mut(col);
            dst.copy_from(&log_tau.column(i));
            dst -= &reference_g;
            sample_names.push(format!("{}@{}", indv, topic).into_boxed_str());
        }
    }
    (signal, sample_names)
}

////////////////////
// Output writers //
////////////////////

/// Bundle of per-(sample / gene / topic) name slices passed to output writers.
pub struct CnvNames<'a> {
    pub sample: &'a [Box<str>],
    pub gene: &'a [Box<str>],
    pub topic: &'a [Box<str>],
}

pub fn write_outputs(
    result: &PerSampleCnv,
    names: &CnvNames<'_>,
    positions: &[GenePosition],
    normals_per_topic: &[Vec<usize>],
    n_indv: usize,
    out_prefix: &str,
) -> anyhow::Result<()> {
    let ordered = &result.genome_order.ordered_indices;
    let ordered_gene_names: Vec<Box<str>> =
        ordered.iter().map(|&i| names.gene[i].clone()).collect();
    let row_meta = (Some(ordered_gene_names.as_slice()), Some("gene"));
    let col_meta = Some(names.sample);

    let g = ordered.len();
    let n = names.sample.len();
    let mut states = DMatrix::<f32>::zeros(g, n);
    for (s, path) in result.viterbi_paths.iter().enumerate() {
        for (gi, &v) in path.iter().enumerate() {
            states[(gi, s)] = v as f32;
        }
    }
    states.to_parquet_with_names(
        &format!("{}.cnv.states.parquet", out_prefix),
        row_meta,
        col_meta,
    )?;
    result.cn_score.to_parquet_with_names(
        &format!("{}.cnv.cn_score.parquet", out_prefix),
        row_meta,
        col_meta,
    )?;

    let mut nlines: Vec<Box<str>> = vec!["topic\tsample\tindv_idx".into()];
    for (k, normals) in normals_per_topic.iter().enumerate() {
        let topic = names.topic.get(k).map(|s| s.as_ref()).unwrap_or("topic");
        for &flat in normals {
            let sample = names.sample.get(flat).map(|s| s.as_ref()).unwrap_or("");
            nlines.push(format!("{}\t{}\t{}", topic, sample, flat % n_indv).into());
        }
    }
    write_lines(
        &nlines,
        &format!("{}.cnv.normal_samples.tsv.gz", out_prefix),
    )?;

    let bed_path = format!("{}.cnv.segments.bed.gz", out_prefix);
    write_segments_bed(result, names.sample, positions, &bed_path)?;

    info!(
        "CNV outputs: {pref}.cnv.states.parquet, {pref}.cnv.cn_score.parquet, {bed}, {pref}.cnv.normal_samples.tsv.gz",
        pref = out_prefix,
        bed = bed_path
    );
    Ok(())
}

/// Walk each sample's Viterbi path per chromosome, emit non-neutral runs as
/// BED6 sorted by (chrom, start). Confidence (BED `score`) uses the mean of
/// `|cn_score|` across the run, scaled to [0, 1000].
fn write_segments_bed(
    result: &PerSampleCnv,
    sample_names: &[Box<str>],
    positions: &[GenePosition],
    bed_path: &str,
) -> anyhow::Result<()> {
    let pos_by_idx: HashMap<usize, &GenePosition> =
        positions.iter().map(|p| (p.gene_idx, p)).collect();
    let ordered = &result.genome_order.ordered_indices;
    let chr_bounds = &result.genome_order.chr_boundaries;
    let neutral_state = result.n_states / 2;

    let mut bed_rows: Vec<(Box<str>, u64, u64, String, u32)> = Vec::new();
    for (s, viterbi) in result.viterbi_paths.iter().enumerate() {
        let sample = sample_names.get(s).map(|n| n.as_ref()).unwrap_or("sample");
        for (chr, chr_start, chr_end) in chr_bounds {
            if chr_start >= chr_end {
                continue;
            }
            let mut seg_start = *chr_start;
            for og in (chr_start + 1)..=*chr_end {
                let seg_state = viterbi[seg_start];
                let same = og < *chr_end && viterbi[og] == seg_state;
                if same {
                    continue;
                }
                if seg_state != neutral_state {
                    let start_pos = pos_by_idx
                        .get(&ordered[seg_start])
                        .map(|p| p.position)
                        .unwrap_or(0);
                    let end_pos = pos_by_idx
                        .get(&ordered[og - 1])
                        .map(|p| p.position)
                        .unwrap_or(0);
                    let conf: f32 = (seg_start..og)
                        .map(|ogg| result.cn_score[(ogg, s)].abs())
                        .sum::<f32>()
                        / (og - seg_start) as f32;
                    let name = format!(
                        "{}|state={}|n={}|p={:.3}",
                        sample,
                        seg_state,
                        og - seg_start,
                        conf
                    );
                    let score = (conf * 1000.0).round().clamp(0.0, 1000.0) as u32;
                    bed_rows.push((
                        chr.clone(),
                        start_pos,
                        end_pos.max(start_pos + 1),
                        name,
                        score,
                    ));
                }
                seg_start = og;
            }
        }
    }
    bed_rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

    use rust_htslib::bgzf::Writer as BWriter;
    use std::io::Write;
    let mut w = BWriter::from_path(bed_path)?;
    for (chr, start, end, name, score) in &bed_rows {
        writeln!(w, "{}\t{}\t{}\t{}\t{}\t.", chr, start, end, name, score)?;
    }
    Ok(())
}

/////////////////////
// DEG concordance //
/////////////////////

pub struct DegConcordance {
    pub r: Vec<f32>,
    pub p: Vec<f32>,
    pub state: Vec<i32>,
}

pub fn compute_deg_concordance(
    signal: &DMatrix<f32>,
    result: &PerSampleCnv,
    n_genes: usize,
) -> DegConcordance {
    let n_samples = signal.ncols();
    assert_eq!(signal.nrows(), n_genes);
    let ordered = &result.genome_order.ordered_indices;

    let mut gene_to_ord = vec![usize::MAX; n_genes];
    for (op, &gi) in ordered.iter().enumerate() {
        if gi < n_genes {
            gene_to_ord[gi] = op;
        }
    }

    let g_ord = ordered.len();
    let n_states = result.n_states.max(1);
    let modal_state_per_op: Vec<i32> = (0..g_ord)
        .into_par_iter()
        .map(|op| modal_state_at(&result.viterbi_paths, op, n_states))
        .collect();
    let n_minus_3 = ((n_samples as f32) - 3.0).max(1.0);

    let ((r_out, p_out), state_out): ((Vec<f32>, Vec<f32>), Vec<i32>) = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let op = gene_to_ord[g];
            if op == usize::MAX {
                return ((f32::NAN, f32::NAN), -1i32);
            }
            let x: Vec<f32> = (0..n_samples).map(|s| signal[(g, s)]).collect();
            let y: Vec<f32> = (0..n_samples).map(|s| result.cn_score[(op, s)]).collect();
            let r = pearson(&x, &y);
            let rc = r.clamp(-0.999_999, 0.999_999);
            let z = 0.5 * ((1.0 + rc) / (1.0 - rc)).ln() * n_minus_3.sqrt();
            ((r, z_to_pvalue(z)), modal_state_per_op[op])
        })
        .unzip();

    DegConcordance {
        r: r_out,
        p: p_out,
        state: state_out,
    }
}

fn pearson(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mx = x.iter().sum::<f32>() / n;
    let my = y.iter().sum::<f32>() / n;
    let (mut num, mut dx, mut dy) = (0f32, 0f32, 0f32);
    for (a, b) in x.iter().zip(y.iter()) {
        let xv = a - mx;
        let yv = b - my;
        num += xv * yv;
        dx += xv * xv;
        dy += yv * yv;
    }
    (num / (dx * dy).sqrt().max(1e-12)).clamp(-1.0, 1.0)
}

///////////////////////////
// Top-level convenience //
///////////////////////////

pub fn run_cnv_calling(
    args: &CnvArgs,
    parameters: &[CocoaGammaOut],
    indv_names: &[Box<str>],
    topic_names: &[Box<str>],
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<Option<(PerSampleCnv, DMatrix<f32>)>> {
    if !args.enabled() {
        return Ok(None);
    }
    let positions = match load_gene_positions(args, gene_names)? {
        Some(p) if !p.is_empty() => p,
        _ => {
            warn!("CNV: no gene positions resolved — skipping CNV calling");
            return Ok(None);
        }
    };

    let n_indv = parameters[0].exposure.posterior_log_mean().ncols();
    let n_topics = parameters.len();
    let n_passes = args.cnv_iter_ref.max(1);
    let cfg = args.per_sample_config();

    let mut refs_local: Vec<Vec<usize>> = (0..n_topics)
        .map(|k| {
            detect_normal_samples(parameters[k].exposure.posterior_log_mean(), CNV_NORMAL_FRAC)
        })
        .collect();
    let mut signal = DMatrix::<f32>::zeros(0, 0);
    let mut sample_names: Vec<Box<str>> = Vec::new();
    let mut result: Option<PerSampleCnv> = None;

    for pass in 1..=n_passes {
        if let Some(prev) = &result {
            refs_local = cluster_reference_from_cn_score(&prev.cn_score, n_topics, n_indv);
        }
        let (sig, names) =
            build_cnv_signal_with_refs(parameters, indv_names, topic_names, &refs_local);
        info!(
            "CNV pass {}/{}: {} genes × {} samples; {} refs per topic",
            pass,
            n_passes,
            sig.nrows(),
            sig.ncols(),
            refs_local.first().map(|v| v.len()).unwrap_or(0)
        );
        result = Some(call_per_sample_cnv(
            &sig, &positions, n_topics, n_indv, &cfg,
        )?);
        signal = sig;
        sample_names = names;
    }
    let result = result.expect("at least one pass");
    let normals_per_topic = topic_local_to_flat(&refs_local, n_indv);

    write_outputs(
        &result,
        &CnvNames {
            sample: &sample_names,
            gene: gene_names,
            topic: topic_names,
        },
        &positions,
        &normals_per_topic,
        n_indv,
        out_prefix,
    )?;

    Ok(Some((result, signal)))
}
