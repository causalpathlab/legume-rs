use anyhow::Result;
use clap::Args;
use log::info;
use matrix_util::common_io::{mkdir_parent, open_buf_writer};
use matrix_util::traits::IoOps;
use std::io::Write;

use auxiliary_data::feature_names::FeatureNameKindArg;
use fagioli::eqtl::{
    classify_states, read_qtl_files, select_top_variants, specificity_rows, train, ubiquity_index,
    EqtlFit, EqtlModelConfig, QtlColumns, SpecificityRow,
};
use fagioli::io::results::write_parameters;
use fagioli::sgvb::ComputeDevice;

#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Embed eQTL summary statistics as a variant-gene-context hyperedge.

Every (variant, gene, cell type) triple scores as a three-way product of
embeddings, sum_h u[j,h] v[g,h] c[k,h]. A cell type is a gate over programs:
a variant-gene pair is expressed there only where their shared programs
overlap that cell type's. Two pseudo-contexts are special values of the
gate rather than special cases in the code. The first is a fixed all-ones
gate named 'ubiquitous', where the score collapses to the context-free
anchor <u,v>. The second is an 'empty' candidate scoring exactly zero,
which every real candidate must beat.

Training is InfoNCE over one positive, a few negatives, and empty.
A negative corrupts ONE slot of the hyperedge, the cell type or the gene.
Negatives come only from cells where absence was CERTIFIED: not detected,
and powered enough to have seen the effect. Cells that were merely
unpowered are sampled in neither class, so the fit cannot learn
statistical power in place of regulatory specificity.

Writes:
  {prefix}.variant_embedding.parquet   variant loadings u, one row per variant
  {prefix}.gene_embedding.parquet      gene loadings v, one row per gene
  {prefix}.context_embedding.parquet   context gates c, including ubiquitous
  {prefix}.specificity.tsv.gz          anchor, per-context scores, ubiquity index
  {prefix}.parameters.json             every setting, the state census, the diagnostics"
)]
pub struct EmbedEqtlArgs {
    // ── Input ────────────────────────────────────────────────────────────
    #[arg(
        long,
        num_args = 1..,
        required = true,
        help_heading = "Input",
        help = "eQTL summary TSV(.gz) files, one row per (variant, gene, celltype)",
        long_help = "Long-format eQTL summary statistics, one or more TSV(.gz) paths.\n\
                     \n\
                     Every file carries a header.\n\
                     Tab, comma, semicolon and space delimiters all read.\n\
                     The header picks the one that splits it widest.\n\
                     A leading '#' on the first header column is tolerated.\n\
                     \n\
                     The --col-* flags name the columns.\n\
                     A flag left at its default also accepts common aliases.\n\
                     So GTEx and eQTL Catalogue headers need no flags.\n\
                     Setting a flag turns its aliases off.\n\
                     A name you asked for and did not get is an error.\n\
                     \n\
                     Multi-member gzip is read whole.\n\
                     Unreadable or truncated files are skipped with a warning.\n\
                     The count reaches parameters.json.\n\
                     Genes and variants may span files.\n\
                     A variant is keyed by its canonical locus.\n\
                     The same position is therefore one entity everywhere."
    )]
    pub qtl_files: Vec<String>,

    #[arg(
        long,
        default_value = "gene",
        help_heading = "Input columns",
        help = "Column with the gene identifier"
    )]
    pub col_gene: String,

    #[arg(
        long,
        default_value = "celltype",
        help_heading = "Input columns",
        help = "Column with the cell type label"
    )]
    pub col_celltype: String,

    #[arg(
        long,
        default_value = "chromosome",
        help_heading = "Input columns",
        help = "Column with the variant chromosome",
        long_help = "Column with the variant chromosome.\n\
                     \n\
                     Chromosome and position only name the variant;\n\
                     no genomic coordinate is interpreted."
    )]
    pub col_chromosome: String,

    #[arg(
        long,
        default_value = "physical.pos",
        help_heading = "Input columns",
        help = "Column with the variant position"
    )]
    pub col_position: String,

    #[arg(
        long,
        default_value = "beta",
        help_heading = "Input columns",
        help = "Column with the effect size"
    )]
    pub col_beta: String,

    #[arg(
        long,
        default_value = "se",
        help_heading = "Input columns",
        help = "Column with the standard error of the effect"
    )]
    pub col_se: String,

    #[arg(
        long,
        default_value = "alpha",
        help_heading = "Input columns",
        help = "Column with a PIP-like row weight (optional in the input)",
        long_help = "Column carrying any nonnegative per-row weight, PIP-like.\n\
                     \n\
                     It only breaks ties between duplicate rows.\n\
                     A duplicate repeats one variant, gene and cell type.\n\
                     The larger weight wins.\n\
                     Files without this column lose nothing else."
    )]
    pub col_pip: String,

    // ── Filtering ────────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "5",
        help_heading = "Filtering",
        help = "Variants kept per gene",
        long_help = "Variants kept per gene.\n\
                     They are ranked by precision-weighted mean chi-square.\n\
                     \n\
                     The ranking runs ONCE per gene.\n\
                     It never runs per gene and cell type.\n\
                     A per-cell-type ranking compares different variants.\n\
                     That destroys the contrast the model must learn.\n\
                     \n\
                     Every gene a selected variant was tested against is kept.\n\
                     Not only the gene it was selected for.\n\
                     Those other pairs are the gene-swap negative pool."
    )]
    pub top_k: usize,

    #[arg(
        long,
        default_value = "4.0",
        help_heading = "Filtering",
        help = "Absolute z at which a cell counts as an edge",
        long_help = "Detection threshold on |beta|/se.\n\
                     \n\
                     Cells at or above it are edges, i.e. the positives.\n\
                     A cell below it is certified absent only when powered.\n\
                     Powered means able to see the pair's reference effect.\n\
                     Everything else is unknown and enters neither class."
    )]
    pub detect_z: f64,

    #[arg(
        long,
        default_value = "1",
        help_heading = "Filtering",
        help = "Drop genes observed in fewer cell types than this",
        long_help = "Minimum number of cell types a gene must be observed in.\n\
                     \n\
                     One keeps every gene, as the validated prototype did.\n\
                     Raise it to drop genes measured too narrowly.\n\
                     Such genes cannot carry a specificity contrast."
    )]
    pub min_celltypes: usize,

    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help_heading = "Input columns",
        help = "Per-name canonicalization rule for the gene column",
        long_help = "How gene names are canonicalized.\n\
                     This is the same rule senna applies.\n\
                     \n\
                     `auto` uses the gene rule below, which suits gene columns.\n\
                     \n\
                     `exact` matches strings strictly.\n\
                     \n\
                     `gene` also aliases each `_`-split component.\n\
                     So `ENSG000_SYM1` and `SYM1` reach the same gene.\n\
                     \n\
                     `locus`, `locus-overlap` and `mixed` are the interval rules.\n\
                     Use one when the gene column carries coordinates.\n\
                     \n\
                     Every rule strips the Ensembl annotation version.\n\
                     So `ENSG00000000001.17` and `.12` are one gene.\n\
                     Clone-based symbols keep their dotted suffix.\n\
                     `CLONE1.4` and `CLONE1.1` stay distinct genes.\n\
                     \n\
                     Variants are always canonicalized as loci.\n\
                     So `chr5:1000` and `5:1000` name one variant."
    )]
    pub feature_name_kind: FeatureNameKindArg,

    #[arg(
        long,
        help_heading = "Filtering",
        help = "Drop rows whose standard error exceeds this value",
        long_help = "Upper bound on the standard error of a row.\n\
                     \n\
                     Unset keeps every finite positive SE.\n\
                     Rows with SE <= 0 or non-finite values are always dropped."
    )]
    pub max_se: Option<f64>,

    // ── Model ────────────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "8",
        help_heading = "Model",
        help = "Number of latent programs H",
        long_help = "Latent programs shared by variants, genes, and cell types.\n\
                     \n\
                     A cell type's gate says which programs are active there.\n\
                     So H bounds how many specificity patterns can be told apart.\n\
                     The gate's effective rank is reported.\n\
                     It settling far below H means the data needed fewer."
    )]
    pub embedding_dim: usize,

    #[arg(
        long,
        default_value = "5",
        help_heading = "Model",
        help = "Negatives drawn per positive",
        long_help = "Corrupted candidates drawn per positive.\n\
                     \n\
                     Each replaces ONE slot, the cell type or the gene.\n\
                     Each comes only from certified-absent cells.\n\
                     More negatives lower the gradient variance.\n\
                     Each costs one score evaluation."
    )]
    pub num_negatives: usize,

    #[arg(
        long,
        default_value = "0.001",
        help_heading = "Model",
        help = "Ridge penalty on the embeddings",
        long_help = "Ridge weight on the three embeddings.\n\
                     \n\
                     It reaches only the rows a batch touched.\n\
                     Penalizing every row every step is far worse.\n\
                     That runs thousands of penalty updates per entity.\n\
                     The data supplies only tens.\n\
                     The loss then pins at log(1 + negatives) regardless."
    )]
    pub ridge: f64,

    #[arg(
        long,
        default_value = "0.2",
        help_heading = "Model",
        help = "Fraction of real-context edges held out of training",
        long_help = "Fraction of real-cell-type edges hidden from training.\n\
                     \n\
                     They are scored afterwards against certified absences.\n\
                     That is the reported held-out AUC and AUPRC.\n\
                     Ubiquitous-context edges are never held out."
    )]
    pub holdout_frac: f64,

    #[arg(
        long,
        default_value_t = false,
        help_heading = "Model",
        help = "Permute the edge/absent labels before training",
        long_help = "Negative control: permute the edge and absent labels.\n\
                     \n\
                     Nothing is left to learn.\n\
                     So the held-out AUC must fall to about one half.\n\
                     The gate's effective rank must rise toward H.\n\
                     Run it whenever the real numbers need a reference."
    )]
    pub shuffle_control: bool,

    // ── Optimization ─────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "0.01",
        help_heading = "Optimization",
        help = "AdamW learning rate",
        long_help = "Step size for the AdamW optimizer.\n\
                     \n\
                     The objective is contrastive.\n\
                     No single quantity is minimised to convergence.\n\
                     Watch the reported loss rather than expect a plateau."
    )]
    pub learning_rate: f64,

    #[arg(
        long,
        default_value = "4000",
        help_heading = "Optimization",
        help = "Number of gradient steps",
        long_help = "Minibatch steps taken over the positives.\n\
                     \n\
                     Cost grows linearly in it.\n\
                     So does how far the embeddings travel from their start."
    )]
    pub num_iterations: usize,

    #[arg(
        long,
        default_value = "256",
        help_heading = "Optimization",
        help = "Positives per gradient step",
        long_help = "Positives drawn per step.\n\
                     \n\
                     A VARIANT is sampled first, one of its positives second.\n\
                     So a variant active in thirty cell types does not outweigh\n\
                     one active in a single cell type.\n\
                     That is exactly the class the model exists to distinguish."
    )]
    pub batch_size: usize,

    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help_heading = "Optimization",
        help = "Hardware device: cpu, cuda, or metal",
        long_help = "Device the fit runs on.\n\
                     \n\
                     The embeddings are small and the batches are indexed\n\
                     lookups, so cpu is usually enough.\n\
                     cuda and metal require the matching build feature."
    )]
    pub device: ComputeDevice,

    #[arg(
        long,
        default_value_t = 0,
        help_heading = "Optimization",
        help = "GPU device index",
        long_help = "Which GPU to use when --device is cuda or metal.\n\
                     Ignored on cpu."
    )]
    pub device_no: usize,

    // ── Misc ─────────────────────────────────────────────────────────────
    #[arg(long, default_value = "42", help = "Random seed for reproducibility")]
    pub seed: u64,

    #[arg(
        short,
        long,
        help = "Output file prefix",
        long_help = "Prefix for every file this run writes.\n\
                     \n\
                     Parent directories are created if they do not exist."
    )]
    pub output: Box<str>,
}

pub fn embed_eqtl(args: &EmbedEqtlArgs) -> Result<()> {
    mkdir_parent(&args.output)?;
    info!("Starting embed-eqtl");

    // ── Step 1: Read ─────────────────────────────────────────────────────
    let cols = QtlColumns {
        gene: args.col_gene.clone(),
        celltype: args.col_celltype.clone(),
        chromosome: args.col_chromosome.clone(),
        position: args.col_position.clone(),
        beta: args.col_beta.clone(),
        se: args.col_se.clone(),
        pip: args.col_pip.clone(),
    };
    let name_kind = args.feature_name_kind.resolve_or_gene();
    let data = read_qtl_files(
        &args.qtl_files,
        &cols,
        &name_kind,
        args.min_celltypes,
        args.max_se,
    )?;

    // ── Step 2: Top-K variants per gene ──────────────────────────────────
    anyhow::ensure!(args.top_k >= 1, "--top-k must be at least 1");
    let selection = select_top_variants(&data, args.top_k);

    // ── Step 3: Ubiquitous context and the three-state evidence ──────────
    let evidence = classify_states(&selection, &data.celltypes, args.detect_z as f32)?;
    let ubiquity = ubiquity_index(&evidence);

    // ── Step 4: Fit ──────────────────────────────────────────────────────
    let device = args.device.to_device(args.device_no)?;
    let config = EqtlModelConfig {
        embedding_dim: args.embedding_dim,
        num_negatives: args.num_negatives,
        num_iterations: args.num_iterations,
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        ridge: args.ridge,
        holdout_frac: args.holdout_frac,
        shuffle_control: args.shuffle_control,
        seed: args.seed,
    };
    let fit = train(&evidence, &data.variants, &data.genes, &config, &device)?;

    // ── Step 5: Write ────────────────────────────────────────────────────
    let h_names: Vec<Box<str>> = (0..args.embedding_dim)
        .map(|h| Box::from(format!("h{h}")))
        .collect();

    for (matrix, entity, map) in [
        (&fit.u, "variant", &fit.variants),
        (&fit.v, "gene", &fit.genes),
        (&fit.c, "context", &fit.contexts),
    ] {
        matrix.to_parquet_with_names(
            &format!("{}.{}_embedding.parquet", args.output, entity),
            (Some(&map.names), Some(entity)),
            Some(&h_names),
        )?;
    }

    let rows = specificity_rows(&evidence, &ubiquity, &fit);
    let n_written = write_specificity(&format!("{}.specificity.tsv.gz", args.output), &rows, &fit)?;

    let input = serde_json::json!({
        "n_files_read": data.n_files_read,
        "n_files_skipped": data.n_files_skipped,
        "n_rows_read": data.n_rows(),
        "n_rows_dropped": data.n_rows_dropped,
        "n_rows_duplicate": data.n_rows_duplicate,
        "n_genes_dropped_min_celltypes": data.n_genes_dropped,
        "n_genes": data.genes.len(),
        "n_variants": data.variants.len(),
        "n_celltypes": data.celltypes.len(),
    });
    let selection_summary = serde_json::json!({
        "n_selected_variants": selection.n_selected_variants,
        "n_pairs_kept": selection.pairs.len(),
        "n_pairs_dropped": selection.n_pairs_dropped,
        "n_rows_kept": selection.n_rows,
    });
    let state_census = serde_json::json!({
        "edge": evidence.n_edge,
        "certified_absent": evidence.n_certified_absent,
        "unknown": evidence.n_unknown,
        "n_cells": evidence.rows.len(),
        "n_pairs": evidence.pairs.len(),
        "n_contexts": evidence.contexts.len(),
    });
    // Serialized whole rather than field by field: a diagnostic added to
    // FitMetrics reaches parameters.json without a second edit here.
    let mut diagnostics = serde_json::to_value(&fit.metrics)?;
    diagnostics["n_specificity_rows"] = serde_json::json!(n_written);

    let params = serde_json::json!({
        "command": "embed-eqtl",
        "qtl_files": args.qtl_files,
        "num_files": args.qtl_files.len(),
        "col_gene": args.col_gene,
        "col_celltype": args.col_celltype,
        "col_chromosome": args.col_chromosome,
        "col_position": args.col_position,
        "col_beta": args.col_beta,
        "col_se": args.col_se,
        "col_pip": args.col_pip,
        "top_k": args.top_k,
        "detect_z": args.detect_z,
        "min_celltypes": args.min_celltypes,
        "max_se": args.max_se,
        "embedding_dim": args.embedding_dim,
        "num_negatives": args.num_negatives,
        "ridge": args.ridge,
        "holdout_frac": args.holdout_frac,
        "shuffle_control": args.shuffle_control,
        "learning_rate": args.learning_rate,
        "num_iterations": args.num_iterations,
        "batch_size": args.batch_size,
        "device": format!("{:?}", args.device),
        "device_no": args.device_no,
        "seed": args.seed,
        "input": input,
        "selection": selection_summary,
        "state_census": state_census,
        "diagnostics": diagnostics,
    });
    write_parameters(&format!("{}.parameters.json", args.output), &params)?;

    info!("embed-eqtl completed successfully");
    Ok(())
}

/// Format the per-pair readout as TSV: one row per (variant, gene), with one
/// score column per real context.
fn write_specificity(path: &str, rows: &[SpecificityRow], fit: &EqtlFit) -> Result<usize> {
    let mut w = open_buf_writer(path)?;
    write!(
        w,
        "variant\tgene\tanchor\tbest_context\tbest_score\tubiquity\tn_powered\tn_edge\tn_unknown"
    )?;
    for k in fit.real_contexts() {
        write!(w, "\t{}", fit.contexts.names[k])?;
    }
    writeln!(w)?;

    for row in rows {
        write!(w, "{}\t{}\t{:.6}\t", row.variant, row.gene, row.anchor)?;
        match (&row.best_context, row.best_score) {
            (Some(name), Some(score)) => write!(w, "{}\t{:.6}", name, score)?,
            _ => write!(w, "NA\tNA")?,
        }
        match row.ubiquity {
            Some(u) => write!(w, "\t{:.4}", u)?,
            None => write!(w, "\tNA")?,
        }
        write!(w, "\t{}\t{}\t{}", row.n_powered, row.n_edge, row.n_unknown)?;
        for s in &row.scores {
            write!(w, "\t{:.6}", s)?;
        }
        writeln!(w)?;
    }
    w.flush()?;
    info!("Wrote {} specificity rows to {}", rows.len(), path);
    Ok(rows.len())
}
