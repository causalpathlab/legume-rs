//! Entry point for `faba annotate` — marker-set cell-type annotation of a
//! `faba gem` run.
//!
//! A thin faba front-end over the shared, model-agnostic **term-ORA** core
//! [`graph_embedding_util::type_annotation::annotate_embeddings_ora`] (Euclidean
//! nearest-centroid assignment → distance-outlier QC → Leiden clustering →
//! cluster×term hypergeometric over-representation, permutation-calibrated →
//! optional TreeBH Cell-Ontology calling). It is the embedding-grounded twin of
//! `senna annotate-by-projection`, reading gem's parquet outputs by prefix (faba
//! has no run manifest).
//!
//! gem carries two gene programs — the spliced/mature identity `β_g` and the
//! nascent splice offset `δ_g` — with two matching cell readouts — the latent `θ`
//! and the velocity increment. Annotation runs per track, one side at a time. The
//! `spliced` track pairs gene `β_g` (`{from}.beta_dictionary.parquet`) with cell `θ`
//! (`{from}.cell_embedding.parquet`) → `{out}.spliced.*`; the `velocity` track pairs gene
//! `δ_g` (`{from}.delta_dictionary.parquet`) with cell velocity
//! (`{from}.velocity.parquet`) → `{out}.velocity.*`.
//!
//! `--track both` (default) runs both; the velocity pass is skipped with a warning
//! when its inputs are absent (a spliced-only gem run, or `--delta-l2 0` on data
//! with no unspliced rows).

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use log::{info, warn};
use std::path::Path;

use graph_embedding_util::type_annotation::{
    annotate_embeddings_ora, InputEmbeddings, TermOraConfig,
};
use matrix_util::common_io::mkdir_parent;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

/// Which gem program(s) to annotate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Track {
    /// Spliced/mature identity: gene β_g vs cell latent θ.
    Spliced,
    /// Nascent splice offset: gene δ_g vs cell velocity increment.
    Velocity,
    /// Both tracks (velocity skipped with a warning if its inputs are missing).
    Both,
}

#[derive(Args, Debug)]
pub struct AnnotateArgs {
    #[arg(
        long,
        short = 'f',
        help = "gem output prefix (the `-o` value passed to `faba gem`)"
    )]
    pub from: Box<str>,

    #[arg(
        long,
        short = 'm',
        help = "Marker TSV: `gene<TAB>celltype` per line (tab/comma/space delimited)"
    )]
    pub markers: Box<str>,

    #[arg(long, short = 'o', help = "Output prefix (default: the gem prefix)")]
    pub out: Option<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value_t = Track::Both,
        help = "Which gem program(s) to annotate"
    )]
    pub track: Track,

    #[arg(
        long,
        default_value_t = 30,
        help = "k for the cosine cell kNN graph fed to Leiden clustering"
    )]
    pub knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Leiden modularity resolution (higher → more, finer clusters)"
    )]
    pub resolution: f64,

    #[arg(
        long,
        default_value_t = 42,
        help = "RNG seed (clustering + permutation null)"
    )]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = 500,
        help = "Permutation draws calibrating the over-representation statistic"
    )]
    pub num_perm: usize,

    #[arg(
        long = "no-assign-qc",
        help = "Disable pruning of high-distance cell→term assignments"
    )]
    pub no_assign_qc: bool,

    #[arg(
        long,
        default_value_t = 2.5,
        help = "MAD multiplier for the assignment-distance outlier gate"
    )]
    pub assign_mad: f64,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "FDR α for the cluster call + Q sparsity (BH on the permutation p)"
    )]
    pub fdr_alpha: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Softmax temperature for the row-normalized Q over significant terms"
    )]
    pub q_temperature: f32,

    #[arg(
        long,
        help = "Disable IDF down-weighting of markers shared across many types"
    )]
    pub no_idf: bool,

    #[arg(
        long,
        help = "Cell Ontology OBO path — runs the TreeBH ontology layer (needs --label-cl)"
    )]
    pub obo: Option<Box<str>>,

    #[arg(long, help = "Curated `label<TAB>CL:id` map (paired with --obo)")]
    pub label_cl: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 0.1,
        help = "TreeBH per-level selective-FDR target"
    )]
    pub ontology_fdr_q: f64,

    #[arg(
        long,
        help = "Benjamini–Yekutieli within ontology families (any dependence)"
    )]
    pub ontology_by: bool,
}

pub fn run_annotate(args: &AnnotateArgs) -> Result<()> {
    let prefix = args.from.as_ref();
    let out = args.out.as_deref().unwrap_or(prefix).to_string();
    mkdir_parent(&out)?;

    let cfg = TermOraConfig {
        knn: args.knn,
        resolution: args.resolution,
        seed: args.seed,
        n_perm: args.num_perm,
        assign_qc: !args.no_assign_qc,
        assign_mad: args.assign_mad,
        fdr_alpha: args.fdr_alpha,
        q_temperature: args.q_temperature,
        obo: args.obo.as_deref().map(str::to_owned),
        label_cl: args.label_cl.as_deref().map(str::to_owned),
        ontology_fdr_q: args.ontology_fdr_q,
        ontology_by: args.ontology_by,
    };

    let want_spliced = matches!(args.track, Track::Spliced | Track::Both);
    let want_velocity = matches!(args.track, Track::Velocity | Track::Both);

    if want_spliced {
        annotate_track(
            prefix,
            &out,
            args,
            &cfg,
            &TrackSpec {
                gene_file: "beta_dictionary.parquet",
                cell_file: "cell_embedding.parquet",
                tag: "spliced",
            },
        )?;
    }

    if want_velocity {
        let gene = format!("{prefix}.delta_dictionary.parquet");
        let cell = format!("{prefix}.velocity.parquet");
        if Path::new(&gene).exists() && Path::new(&cell).exists() {
            annotate_track(
                prefix,
                &out,
                args,
                &cfg,
                &TrackSpec {
                    gene_file: "delta_dictionary.parquet",
                    cell_file: "velocity.parquet",
                    tag: "velocity",
                },
            )?;
        } else {
            warn!(
                "velocity track skipped: missing {gene} and/or {cell} \
                 (run `faba gem` on spliced+unspliced counts to produce them)"
            );
            if args.track == Track::Velocity {
                anyhow::bail!("--track velocity requested but δ_g / velocity outputs are absent");
            }
        }
    }

    info!("faba annotate complete (prefix '{out}')");
    Ok(())
}

/// The gem parquet suffixes + output tag identifying one annotation track.
struct TrackSpec {
    /// Gene-embedding parquet suffix (`{from}.{gene_file}`): β_g or δ_g.
    gene_file: &'static str,
    /// Cell-embedding parquet suffix (`{from}.{cell_file}`): latent θ or velocity.
    cell_file: &'static str,
    /// Output tag: outputs land at `{out}.{tag}.*`.
    tag: &'static str,
}

/// Annotate one track: load the gene embedding + cell embedding parquet, hand them
/// to the shared term-ORA core, and write `{out}.{tag}.*`.
fn annotate_track(
    prefix: &str,
    out: &str,
    args: &AnnotateArgs,
    cfg: &TermOraConfig,
    spec: &TrackSpec,
) -> Result<()> {
    let gene_path = format!("{prefix}.{}", spec.gene_file);
    let cell_path = format!("{prefix}.{}", spec.cell_file);
    let feat = DMatrix::<f32>::from_parquet(&gene_path)
        .with_context(|| format!("reading gene embedding {gene_path}"))?;
    let cell = DMatrix::<f32>::from_parquet(&cell_path)
        .with_context(|| format!("reading cell embedding {cell_path}"))?;
    info!(
        "annotate track '{}': genes [{} × {}], cells [{} × {}]",
        spec.tag,
        feat.mat.nrows(),
        feat.mat.ncols(),
        cell.mat.nrows(),
        cell.mat.ncols()
    );
    annotate_embeddings_ora(
        &InputEmbeddings {
            feature_emb: &feat.mat,
            gene_names: &feat.rows,
            cell_emb: &cell.mat,
            cell_names: &cell.rows,
        },
        &args.markers,
        &format!("{out}.{}", spec.tag),
        !args.no_idf,
        cfg,
    )
}
