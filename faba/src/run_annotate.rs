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
//! The gene side is gem's **co-embedded feature vectors**
//! (`{from}.feature_embedding.parquet`), *not* the `β_g` dictionary — a Euclidean
//! nearest-centroid call is only meaningful when genes and cells share a metric
//! space, and β does not share one with θ. See [`crate::gem_gene_embedding`] for the
//! measurements. Those rows are keyed by feature (`{gene}/count/{spliced,unspliced}`),
//! so each track selects its own modality and re-keys by gene.
//!
//! Annotation runs per track, one side at a time. The `spliced` track pairs the
//! spliced feature rows with cell `θ` (`{from}.cell_embedding.parquet`) →
//! `{out}.spliced.*`; the `velocity` track pairs the unspliced rows with the cell
//! velocity increment (`{from}.velocity.parquet`) → `{out}.velocity.*`.
//!
//! `--track both` (default) runs both; the velocity pass is skipped with a warning
//! when its inputs are absent (a spliced-only gem run, or `--delta-l2 0` on data
//! with no unspliced rows).

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use log::{info, warn};
use std::path::Path;

use crate::gem_gene_embedding::{load_gene_embedding, Modality};
use graph_embedding_util::type_annotation::{
    annotate_embeddings_ora, Abstain, InputEmbeddings, MarkerBootstrapConfig, TermOraConfig,
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

    #[arg(
        long,
        default_value_t = 0,
        help = "Marker-panel permutation null (the BIAS guard). 0 = off; try 200",
        long_help = "Marker-panel permutation null — the BIAS guard.\n\n\
            Puts each type on trial: replace ONLY its markers with the same number of\n\
            random genes (same IDF weights, matched on gene norm, drawn from the live\n\
            marker pool), leave every rival type real, and ask whether its own genes\n\
            place its prototype any better than random ones would.\n\n\
            The bootstrap only measures VARIANCE, so a type whose markers are simply\n\
            wrong comes back perfectly stable and looks like the most confident call in\n\
            the run. This is what catches that.\n\n\
            0 = off; try 200. Writes {out}.panel_null.tsv"
    )]
    pub panel_perm: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Support permutation null: turns label_support into a p-value/FDR. 0 = off",
        long_help = "Support permutation null — calibrates `label_support`.\n\n\
            Shuffles which type each marker gene belongs to (within gene-norm strata, so\n\
            no type's norm profile changes) and re-runs the whole bootstrap, to learn\n\
            what a cell's support looks like when the panel carries no type information.\n\n\
            This replaces an arbitrary bar with a calibrated one. --min-support 0.5 is\n\
            not scale-free: with C types, chance agreement is 1/C, so 0.5 sits at 3x\n\
            chance on a 6-type panel and 12x on a 24-type one — the same flag is a\n\
            different test on different panels. An FDR means the same thing everywhere.\n\n\
            0 = off; needs the bootstrap. Reuses the bootstrap's cached partitions, so\n\
            the cost is the cheap half of a replicate, not a re-clustering.\n\
            Adds support_p / support_q / null_support to {out}.annot.parquet"
    )]
    pub support_perm: usize,

    #[arg(
        long = "no-bootstrap-markers",
        help = "Turn OFF the stability bootstrap and ship a bare point estimate",
        long_help = "Turn OFF the stability bootstrap and ship a bare point estimate.\n\n\
            The bootstrap is ON by default. Each draw resamples every type's marker panel\n\
            with replacement AND re-derives the clustering; the consensus is what ships.\n\
            So every call carries the fraction of resamples that agreed on it, and a call\n\
            that cannot hold up across them abstains rather than being printed.\n\n\
            Without it, `argmin` over marker centroids always returns something, and\n\
            returns it with no error bar. Measured on cord blood: 28.2% of cells were\n\
            assigned to types the tissue does not contain, against 2.4% with it on"
    )]
    pub no_bootstrap_markers: bool,

    #[arg(
        long,
        default_value_t = 200,
        help = "Bootstrap resamples (0 or --no-bootstrap-markers to disable)"
    )]
    pub n_boot: usize,

    #[arg(
        long,
        help = "Hold the clustering fixed across resamples (weakens the bootstrap)",
        long_help = "Hold the clustering fixed across resamples.\n\n\
            By default each draw re-derives the clustering, so the partition's own\n\
            arbitrariness is absorbed into the support rather than silently trusted.\n\
            The kNN graph is deterministic (so runs reproduce), but Leiden still picks\n\
            among near-equal modularity optima, and a label that flips when the\n\
            partition is re-drawn is not a robust one.\n\n\
            WARNING: with the partition held fixed the bootstrap has little to say —\n\
            measured, NOTHING abstains (0% unassigned) and support's ability to separate\n\
            spurious calls falls from AUC 0.93 to 0.69"
    )]
    pub no_recluster: bool,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Minimum fraction of resamples the top label must win to be called",
        long_help = "Minimum fraction of resamples the top label must win for the cell to\n\
            be called at all.\n\n\
            NOTE this bar is NOT scale-free. With C types, chance agreement is 1/C, so\n\
            0.5 sits at ~3x chance on a 6-type panel and ~12x chance on a 24-type one —\n\
            the same value is a different test on different panels, and their abstention\n\
            rates are not comparable.\n\n\
            --abstain-separable (a sign test) and --support-perm (a calibrated FDR) both\n\
            avoid that"
    )]
    pub min_support: f32,

    #[arg(
        long,
        conflicts_with = "min_support",
        help = "Abstain by a sign test instead of the --min-support threshold",
        long_help = "Abstain by a TEST rather than a threshold.\n\n\
            Keep the top label only if it beat the runner-up by more than resampling\n\
            noise — an exact binomial sign test at --abstain-alpha. Among the m\n\
            replicates that chose one of the two leading labels, each is a coin flip if\n\
            the two are equally probable.\n\n\
            No magic number, and unlike --min-support it means the same thing whatever\n\
            the number of types. It resolves more cells, but note it decides WHEN to stay\n\
            silent, not whether a call is right"
    )]
    pub abstain_separable: bool,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "[--abstain-separable] Significance level for the top-vs-runner-up sign test"
    )]
    pub abstain_alpha: f64,

    #[arg(
        long,
        default_value_t = 0.8,
        help = "Coverage of the reported `label_set` (the mixed annotation)",
        long_help = "Coverage of the reported `label_set` — the smallest set of labels\n\
            accounting for this share of the resamples.\n\n\
            A cell that cannot be given ONE label can still be given two, and\n\
            `HSPC/LMPP` is a far better answer than `unassigned`. The distribution is\n\
            already computed by the bootstrap; this stops us throwing it away"
    )]
    pub set_coverage: f32,

    #[arg(
        long,
        default_value_t = 3,
        help = "Largest `label_set` worth printing (a 4-way tie is not an annotation)",
        long_help = "Largest `label_set` worth printing.\n\n\
            `HSPC/LMPP` is an annotation; a four-way tie is not — past a point a set stops\n\
            narrowing anything down and starts laundering \"we don't know\" as though it\n\
            were a finding.\n\n\
            A cell that needs more labels than this to reach --set-coverage is left\n\
            unassigned"
    )]
    pub max_set_size: usize,

    #[arg(
        long,
        default_value_t = 0.0,
        value_name = "FRAC",
        help = "Fail if under this fraction of the marker panel is on the embedding's \
                feature axis (0 = report only)",
        long_help = "Refuse to annotate on a marker panel the embedding mostly never saw.\n\n\
            `faba gem` writes only its TRAINED feature rows, so a marker that missed the\n\
            `--n-hvg` cut is not down-weighted — it is ABSENT, and it silently drops out of\n\
            the panel. A cell type that entered with 20 markers and scores on 1 still gets a\n\
            confident-looking call, and nothing in the output distinguishes it from a\n\
            well-supported one.\n\n\
            The coverage is always reported, and a type keeping under half its panel always\n\
            warns. This makes it fatal instead. The fix when it fires is to widen the axis\n\
            (raise `--n-hvg`) or to force the panel into training — `faba gem --markers\n\
            <the same file>`."
    )]
    pub min_panel_coverage: f32,
}

pub fn run_annotate(args: &AnnotateArgs) -> Result<()> {
    let prefix = args.from.as_ref();
    let out = args.out.as_deref().unwrap_or(prefix).to_string();
    mkdir_parent(&out)?;

    let cfg = TermOraConfig {
        min_panel_coverage: args.min_panel_coverage,
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
        panel_perm: args.panel_perm,
        support_perm: args.support_perm,
        bootstrap: (!args.no_bootstrap_markers).then_some(MarkerBootstrapConfig {
            n_boot: args.n_boot,
            abstain: if args.abstain_separable {
                Abstain::Separable(args.abstain_alpha)
            } else {
                Abstain::Support(args.min_support)
            },
            set_coverage: args.set_coverage,
            max_set_size: args.max_set_size,
            recluster: !args.no_recluster,
        }),
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
                modality: Modality::Spliced,
                cell_file: "cell_embedding.parquet",
                tag: "spliced",
            },
        )?;
    }

    if want_velocity {
        let gene = format!("{prefix}.feature_embedding.parquet");
        let cell = format!("{prefix}.velocity.parquet");
        if Path::new(&gene).exists() && Path::new(&cell).exists() {
            annotate_track(
                prefix,
                &out,
                args,
                &cfg,
                &TrackSpec {
                    modality: Modality::Unspliced,
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

/// One annotation track: which of gem's feature-embedding rows to score, against which cell
/// readout.
struct TrackSpec {
    /// Which feature rows carry this track's gene program (see [`Modality`]).
    modality: Modality,
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
    let cell_path = format!("{prefix}.{}", spec.cell_file);
    let feat = load_gene_embedding(prefix, spec.modality)?;
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
