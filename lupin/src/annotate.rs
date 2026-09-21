//! Unified `lupin annotate` — senna enrichment/projection plus pinto-style embedding ORA.

use std::path::Path;

use annotate::{AnnotateArgs, AnnotateOntologyArgs, AnnotateProjectionArgs};
use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use graph_embedding_util::type_annotation::{
    annotate_embeddings_ora, InputEmbeddings, TermOraConfig,
};
use legume_numeric::matrix::common_io::mkdir_parent;
use legume_numeric::matrix::dmatrix_io::DMatrix;
use legume_numeric::matrix::traits::IoOps;
use senna::annotate::{annotate_by_enrichment, annotate_by_projection, annotate_ontology};
use senna::run_manifest::{self, RunKind};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum AnnotateMethod {
    #[default]
    Auto,
    Enrichment,
    Projection,
}

#[derive(Args, Debug)]
pub struct AnnotateCliArgs {
    #[arg(
        long,
        value_enum,
        default_value_t = AnnotateMethod::Auto,
        help = "Annotation backend: enrichment (topic/svd), projection (co-embed), or auto"
    )]
    pub method: AnnotateMethod,

    #[arg(
        long,
        short = 'f',
        help = "Run manifest / embedding prefix (`run.senna.json`, pinto `-o`, or bare prefix)"
    )]
    pub from: Option<Box<str>>,

    #[arg(
        long,
        help = "Feature × D embedding parquet (pinto / explicit projection path)"
    )]
    pub feature_embedding: Option<Box<str>>,

    #[arg(
        long,
        help = "Cell × D embedding parquet (pinto / explicit projection path)"
    )]
    pub cell_embedding: Option<Box<str>>,

    #[arg(
        long,
        short = 'm',
        default_value = "",
        help = "Marker TSV (`gene<TAB>celltype`); omit for ontology-only follow-up on enrichment output"
    )]
    pub markers: Box<str>,

    #[arg(long, short = 'o', help = "Output prefix")]
    pub out: Option<Box<str>>,

    // ── shared clustering / stats ──
    // Default is method-specific when omitted: enrichment 15, projection/ORA 30.
    #[arg(long)]
    pub knn: Option<usize>,
    #[arg(long, default_value_t = 1.0)]
    pub resolution: f64,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long = "num-perm", default_value_t = 500)]
    pub num_perm: usize,
    #[arg(long = "min-markers", default_value_t = 3)]
    pub min_markers: usize,
    #[arg(long = "fdr-alpha", default_value_t = 0.1)]
    pub fdr_alpha: f32,
    #[arg(long = "q-temperature", default_value_t = 1.0)]
    pub q_temperature: f32,
    #[arg(long = "no-clean")]
    pub no_clean: bool,
    #[arg(short = 'v', long)]
    pub verbose: bool,

    // ── enrichment-only ──
    #[arg(long = "clusters")]
    pub clusters: Option<Box<str>>,
    #[arg(long = "num-clusters")]
    pub num_clusters: Option<usize>,
    #[arg(long = "min-cluster-size", default_value_t = 2)]
    pub min_cluster_size: usize,
    #[arg(long = "cluster-seed")]
    pub cluster_seed: Option<u64>,
    #[arg(long = "gaf")]
    pub gaf: Option<Box<str>>,
    #[arg(long = "gmt")]
    pub gmt: Option<Box<str>>,

    // ── projection / ORA ──
    #[arg(long = "no-idf")]
    pub no_idf: bool,
    #[arg(long = "no-assign-qc")]
    pub no_assign_qc: bool,
    #[arg(long = "assign-mad", default_value_t = 2.5)]
    pub assign_mad: f64,

    // ── ontology (inline on annotate, or follow-up without markers) ──
    #[arg(long = "obo")]
    pub obo: Option<Box<str>>,
    #[arg(long = "label-cl")]
    pub label_cl: Option<Box<str>>,
    #[arg(long = "ontology-fdr-q", default_value_t = 0.1)]
    pub ontology_fdr_q: f64,
    #[arg(long = "ontology-by", help = "TreeBH Benjamini–Yekutieli")]
    pub ontology_by: bool,
    #[arg(long = "use-perm-p")]
    pub use_perm_p: bool,
}

pub fn run_annotate(args: &AnnotateCliArgs) -> Result<()> {
    if is_ontology_followup(args) {
        return run_ontology_followup(args);
    }

    let method = resolve_method(args)?;
    match method {
        AnnotateMethod::Enrichment => run_enrichment(args),
        AnnotateMethod::Projection => run_projection(args),
        AnnotateMethod::Auto => unreachable!("resolve_method maps Auto"),
    }
}

fn is_ontology_followup(args: &AnnotateCliArgs) -> bool {
    args.markers.is_empty()
        && args.gaf.is_none()
        && args.gmt.is_none()
        && args.from.is_some()
        && args.obo.is_some()
        && args.label_cl.is_some()
}

fn run_ontology_followup(args: &AnnotateCliArgs) -> Result<()> {
    let from = args
        .from
        .as_ref()
        .context("--from required for ontology follow-up")?;
    let obo = args.obo.as_ref().context("--obo required")?;
    let label_cl = args.label_cl.as_ref().context("--label-cl required")?;
    annotate_ontology(&AnnotateOntologyArgs {
        from: from.clone(),
        label_cl: label_cl.clone(),
        obo: obo.clone(),
        out: args.out.clone(),
        fdr_q: args.ontology_fdr_q,
        by: args.ontology_by,
        use_perm_p: args.use_perm_p,
        verbose: args.verbose,
    })
}

fn resolve_method(args: &AnnotateCliArgs) -> Result<AnnotateMethod> {
    match args.method {
        AnnotateMethod::Enrichment | AnnotateMethod::Projection => Ok(args.method),
        AnnotateMethod::Auto => {
            // Senna manifests first: their feature_embedding.parquet is ρ, not the
            // marker co-embed — never treat those filenames as a pinto ORA input.
            if senna_manifest_prefers_projection(args)? || resolve_embedding_paths(args)?.is_some()
            {
                Ok(AnnotateMethod::Projection)
            } else {
                Ok(AnnotateMethod::Enrichment)
            }
        }
    }
}

/// Explicit `--feature-embedding`/`--cell-embedding`, or `{prefix}.feature_embedding.parquet`
/// + `{prefix}.cell_embedding.parquet` from `--from`.
fn resolve_embedding_paths(args: &AnnotateCliArgs) -> Result<Option<(String, String, String)>> {
    if let (Some(feat), Some(cell)) = (
        args.feature_embedding.as_deref(),
        args.cell_embedding.as_deref(),
    ) {
        let prefix = args
            .out
            .as_deref()
            .or(args.from.as_deref())
            .map(run_manifest::derive_out_prefix)
            .unwrap_or_else(|| "annot".into());
        return Ok(Some((feat.to_string(), cell.to_string(), prefix)));
    }
    let Some(from) = args.from.as_deref() else {
        return Ok(None);
    };
    let prefix = run_manifest::derive_out_prefix(from);
    let feat_path = format!("{prefix}.feature_embedding.parquet");
    let cell_path = format!("{prefix}.cell_embedding.parquet");
    if Path::new(&feat_path).is_file() && Path::new(&cell_path).is_file() {
        Ok(Some((feat_path, cell_path, prefix)))
    } else {
        Ok(None)
    }
}

fn senna_manifest_prefers_projection(args: &AnnotateCliArgs) -> Result<bool> {
    let Some(from) = args.from.as_deref() else {
        return Ok(false);
    };
    let Ok((manifest, _)) = run_manifest::load_for(from) else {
        return Ok(false);
    };
    let coembeds = matches!(
        manifest.kind,
        RunKind::Bge | RunKind::Gem | RunKind::Simba | RunKind::ResolveEmbeddingSpace
    );
    Ok(manifest.outputs.feature_coembedding.is_some()
        || (coembeds && manifest.outputs.feature_embedding.is_some()))
}

fn run_enrichment(args: &AnnotateCliArgs) -> Result<()> {
    let from = args
        .from
        .clone()
        .context("--from is required for enrichment annotation")?;
    anyhow::ensure!(
        !args.markers.is_empty() || args.gaf.is_some() || args.gmt.is_some(),
        "enrichment needs --markers, --gaf, or --gmt"
    );
    annotate_by_enrichment(&build_enrichment_args(args, from))
}

fn run_projection(args: &AnnotateCliArgs) -> Result<()> {
    let explicit = args.feature_embedding.is_some() && args.cell_embedding.is_some();
    let senna_from = args
        .from
        .as_deref()
        .is_some_and(|f| run_manifest::load_for(f).is_ok());

    // Explicit embedding pair → pinto-style ORA. Senna manifests with only `--from`
    // must go through annotate_by_projection (co-embed), even though they also write
    // feature_embedding.parquet + cell_embedding.parquet (ρ and Z).
    if explicit || (!senna_from && resolve_embedding_paths(args)?.is_some()) {
        let (feat, cell, prefix) = resolve_embedding_paths(args)?
            .context("provide --feature-embedding/--cell-embedding or a pinto --from prefix")?;
        return run_embedding_ora(args, &feat, &cell, &prefix);
    }

    let from = args
        .from
        .clone()
        .context("--from is required for senna projection annotation")?;
    anyhow::ensure!(!args.markers.is_empty(), "projection needs --markers");
    annotate_by_projection(&build_projection_args(args, from))
}

fn run_embedding_ora(
    args: &AnnotateCliArgs,
    feat_path: &str,
    cell_path: &str,
    prefix: &str,
) -> Result<()> {
    anyhow::ensure!(!args.markers.is_empty(), "projection needs --markers");

    let feat = DMatrix::<f32>::from_parquet(feat_path)
        .with_context(|| format!("reading feature embedding {feat_path}"))?;
    let cell = DMatrix::<f32>::from_parquet(cell_path)
        .with_context(|| format!("reading cell embedding {cell_path}"))?;

    let out = args.out.as_deref().unwrap_or(prefix).to_string();
    mkdir_parent(&out)?;

    let cfg = TermOraConfig {
        min_panel_coverage: 0.0,
        knn: args.knn.unwrap_or(30),
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
        min_markers: args.min_markers,
        panel_perm: 0,
        support_perm: 0,
        bootstrap: None,
    };

    annotate_embeddings_ora(
        &InputEmbeddings {
            feature_emb: &feat.mat,
            gene_names: &feat.rows,
            cell_emb: &cell.mat,
            cell_names: &cell.rows,
        },
        &args.markers,
        &out,
        !args.no_idf,
        &cfg,
    )
}

fn build_enrichment_args(args: &AnnotateCliArgs, from: Box<str>) -> AnnotateArgs {
    AnnotateArgs {
        from,
        clusters: args.clusters.clone(),
        knn: args.knn.unwrap_or(15),
        resolution: args.resolution,
        num_clusters: args.num_clusters,
        min_cluster_size: args.min_cluster_size,
        cluster_seed: args.cluster_seed,
        markers: args.markers.clone(),
        gaf: args.gaf.clone(),
        gmt: args.gmt.clone(),
        no_iea: false,
        min_gene_set: 15,
        max_gene_set: 500,
        out: args.out.clone(),
        verbose: args.verbose,
        block_size: 1024,
        num_draws: 1000,
        num_perm: args.num_perm,
        min_markers: args.min_markers,
        fdr_alpha: args.fdr_alpha,
        q_temperature: args.q_temperature,
        min_confidence: 0.0,
        seed: args.seed,
        no_clean: args.no_clean,
        preload_data: false,
        no_empirical_specificity: false,
        obo: args.obo.clone(),
        label_cl: args.label_cl.clone(),
        ontology_fdr_q: args.ontology_fdr_q,
        ontology_by: args.ontology_by,
        no_gene_strata: false,
        no_bootstrap_markers: false,
        n_boot: 200,
        boot_num_draws: 100,
        min_support: 0.5,
        abstain_separable: false,
        abstain_alpha: 0.05,
        set_coverage: 0.8,
        max_set_size: 3,
    }
}

fn build_projection_args(args: &AnnotateCliArgs, from: Box<str>) -> AnnotateProjectionArgs {
    AnnotateProjectionArgs {
        from,
        markers: args.markers.clone(),
        out: args.out.clone(),
        knn: args.knn.unwrap_or(30),
        resolution: args.resolution,
        num_perm: args.num_perm,
        seed: args.seed,
        min_markers: args.min_markers,
        no_idf: args.no_idf,
        no_assign_qc: args.no_assign_qc,
        assign_mad: args.assign_mad,
        fdr_alpha: args.fdr_alpha,
        q_temperature: args.q_temperature,
        obo: args.obo.clone(),
        label_cl: args.label_cl.clone(),
        ontology_fdr_q: args.ontology_fdr_q,
        ontology_by: args.ontology_by,
        panel_perm: 0,
        support_perm: 0,
        no_bootstrap_markers: false,
        n_boot: 200,
        no_recluster: false,
        min_support: 0.5,
        abstain_separable: false,
        abstain_alpha: 0.05,
        set_coverage: 0.8,
        max_set_size: 3,
        no_clean: args.no_clean,
        verbose: args.verbose,
    }
}
