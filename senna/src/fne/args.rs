//! `senna fne`'s command-line surface. Training defaults are PBG's own
//! (the settings SIMBA's `pbg_train` uses) at the workspace's embedding
//! dimension, so a bare invocation is the published recipe.

use crate::embed_common::*;

#[derive(Args, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default = "crate::embed_common::clap_defaults")]
pub struct FneArgs {
    #[arg(
        value_delimiter = ',',
        help = "Gene-gene edge list(s) (TSV/CSV; two columns per line, optional weight)",
        long_help = "Zero or more positional paths, comma-separated or space-separated.\n\
                     Each file is whitespace/comma/tab-delimited;\n\
                     every line is a pair of gene names,\n\
                     with an optional third column holding a per-edge weight.\n\
                     Lines starting with `#` are skipped;\n\
                     self-loops are dropped and a repeated pair keeps its largest weight.\n\
                     Every file is its own relation, named `gene:gene/<file stem>`\n\
                     (the file name without its .tsv/.csv/.gz extensions),\n\
                     so BioGRID and STRING can be weighted apart with --relation-weight."
    )]
    pub(crate) networks: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Typed edge list(s): lhs_type, lhs, rhs_type, rhs [, weight]",
        long_help = "Typed edge files, comma-separated or repeated.\n\
                     Every line is `lhs_type <TAB> lhs <TAB> rhs_type <TAB> rhs [<TAB> weight]`,\n\
                     e.g. `gene TP53 term GO:0006915 1.0` or `gene TP53 word apoptosis 0.61`.\n\
                     Rows sharing a type pair form one relation named `lhs_type:rhs_type`,\n\
                     across every file.\n\
                     A relation whose two types coincide is undirected.\n\
                     Names of type `gene` are canonicalised like the positional files;\n\
                     every other type is matched exactly."
    )]
    pub(crate) edges: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Membership file(s), `type=path`: gene <TAB> label rows become gene:<type> edges",
        long_help = "Membership files, `type=path`, comma-separated or repeated,\n\
                     e.g. `cell_type=markers.tsv` or `tf=tf_targets.tsv`.\n\
                     Every row is `gene <TAB> label`, tab or comma delimited;\n\
                     a header and `#` rows are skipped.\n\
                     The labels become nodes of the given type\n\
                     and the rows the relation `gene:<type>/<file stem>`."
    )]
    pub(crate) membership: Vec<Box<str>>,

    #[arg(
        long,
        help = "GO annotations (GAF, .gaf or .gaf.gz); needs --obo",
        long_help = "A GO annotation file.\n\
                     Every gene→term row becomes a gene:term edge,\n\
                     propagated up the ontology (is_a + part_of, the true-path rule) when --obo is given,\n\
                     so a gene annotated to a leaf also links to every ancestor.\n\
                     Terms outside --min/--max-gene-set are dropped."
    )]
    pub(crate) gaf: Option<Box<str>>,

    #[arg(
        long,
        help = "Ontology (OBO) for --gaf propagation, term:term edges and term text",
        long_help = "An OBO ontology (go-basic.obo, cl-basic.obo).\n\
                     Besides propagating --gaf annotations,\n\
                     its hierarchy joins the graph as the relations `term:term/is_a` and `term:term/part_of`\n\
                     over the terms kept,\n\
                     and its names and definitions feed --export-text."
    )]
    pub(crate) obo: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Drop IEA (electronic) annotations from --gaf"
    )]
    pub(crate) no_iea: bool,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Gene-set file(s) (GMT): each set becomes a term node",
        long_help = "MSigDB-style GMT files, comma-separated or repeated.\n\
                     Every line is `term <TAB> description <TAB> gene...`;\n\
                     the set's genes link to the term node in the relation `gene:term/<file stem>`.\n\
                     Sets outside --min/--max-gene-set are dropped;\n\
                     the description feeds --export-text."
    )]
    pub(crate) gmt: Vec<Box<str>>,

    #[arg(
        long,
        default_value_t = 5,
        help = "Smallest gene set (GAF/GMT term) kept"
    )]
    pub(crate) min_gene_set: usize,

    #[arg(
        long,
        default_value_t = 500,
        help = "Largest gene set (GAF/GMT term) kept; 0 = no cap"
    )]
    pub(crate) max_gene_set: usize,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Region-gene link file(s): region <TAB> gene [<TAB> score]",
        long_help = "Genomic region→gene links (eQTL, peak-to-gene, ABC), comma-separated or repeated.\n\
                     A region is `chr:start-end`, `chr_start_end`, or a position `chr:pos` / `chr_pos`;\n\
                     `chr` prefixes are dropped.\n\
                     Each region is tiled onto fixed windows (--region-window)\n\
                     and links its gene from every window it overlaps,\n\
                     in the relation `region:gene/<file stem>`;\n\
                     the score column is the edge weight."
    )]
    pub(crate) region_gene: Vec<Box<str>>,

    #[arg(
        long,
        default_value_t = 5000,
        help = "Window size (bp) the regions are tiled onto; 0 keeps regions as given"
    )]
    pub(crate) region_window: i64,

    #[arg(
        long,
        help = "Write `feature <TAB> type <TAB> name <TAB> text` for every node with text",
        long_help = "Export the text the inputs carry (OBO names and definitions, GMT descriptions)\n\
                     as `feature <TAB> type <TAB> name <TAB> text`,\n\
                     one row per node that has any.\n\
                     This is the input of the text encoder that turns descriptions into gene:word edges."
    )]
    pub(crate) export_text: Option<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Relation weight override(s), `name=weight`",
        long_help = "Loss weight of a relation, `name=weight`, comma-separated or repeated.\n\
                     Names are the ones the run logs,\n\
                     e.g. `gene:gene/biogrid=2`, `gene:term/goa_human=0.5` or `gene:word=0.5`.\n\
                     Every relation defaults to 1."
    )]
    pub(crate) relation_weight: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Passes over a relation's edges per epoch, `name=k`",
        long_help = "How many times a relation's training edges are drawn per epoch, `name=k`,\n\
                     comma-separated or repeated; every relation defaults to 1.\n\
                     Batches are drawn in proportion to the edges left,\n\
                     so a small relation beside a large one (a marker panel beside a PPI)\n\
                     gets few updates per epoch and its nodes stay near their init;\n\
                     repeating it restores its share without changing the loss weight."
    )]
    pub(crate) relation_repeat: Vec<Box<str>>,

    #[arg(
        long,
        default_value_t = 128,
        alias = "dim-embedding",
        help = "Embedding dimension H"
    )]
    pub(crate) embedding_dim: usize,

    #[arg(
        long,
        short = 'i',
        default_value_t = 10,
        help = "Training epochs (PBG: 10)"
    )]
    pub(crate) epochs: usize,

    #[arg(
        long,
        alias = "lr",
        default_value_t = 0.1,
        help = "Row-wise Adagrad learning rate (PBG: 0.1)"
    )]
    pub(crate) learning_rate: f64,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Edges per batch (PBG: 1000)",
        long_help = "Edges per batch. Every batch holds ONE relation,\n\
                     drawn with probability proportional to that relation's remaining edges.\n\
                     One optimizer step per batch."
    )]
    pub(crate) batch_size: usize,

    #[arg(
        long,
        default_value_t = 50,
        help = "Batch negatives, i.e. the chunk size (PBG: 50)",
        long_help = "A batch is cut into chunks of this many positives.\n\
                     Within a chunk every other positive's endpoints are negatives.\n\
                     A positive never competes with itself."
    )]
    pub(crate) num_batch_negs: usize,

    #[arg(
        long,
        default_value_t = 50,
        help = "Uniform negatives per chunk (PBG: 50)",
        long_help = "Random nodes drawn per chunk and shared by its positives,\n\
                     inside the relation's own node types on each side.\n\
                     Both sides are corrupted."
    )]
    pub(crate) num_uniform_negs: usize,

    #[arg(
        long,
        help = "Weight decay; omit for SIMBA's automatic value",
        long_help = "L2 weight decay on the node table.\n\
                     Omit it for SIMBA's `auto_wd`, which scales a reference value by the edge count.\n\
                     Pass 0 to disable."
    )]
    pub(crate) weight_decay: Option<f64>,

    #[arg(
        long,
        default_value_t = 50,
        help = "Draw the weight decay with probability 1/N per batch (PBG: 50)"
    )]
    pub(crate) wd_interval: usize,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "Fraction of each relation's edges held out for the eval loss (PBG: 0.05)",
        long_help = "Edges never trained on, scored with the same loss after every epoch.\n\
                     Held out per relation, so a small relation is never emptied; drawn once.\n\
                     Pass 0 to train on every edge."
    )]
    pub(crate) eval_fraction: f64,

    #[arg(
        long,
        default_value_t = 1,
        hide = true,
        help = "Held-out edges per relation at least, when the fraction is positive"
    )]
    pub(crate) eval_min_per_relation: usize,

    #[arg(
        long,
        default_value_t = '_',
        help = "Delimiter for canonical gene-name matching",
        long_help = "Delimiter for canonical gene-name matching.\n\
                     The last token after splitting on this char is canonical.\n\
                     So `ENSG00000_TGFB1` and `TGFB1` merge into one node."
    )]
    pub(crate) feature_name_delim: char,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable fuzzy gene-name matching (use exact names)"
    )]
    pub(crate) feature_name_exact: bool,

    #[arg(long, default_value_t = 1, help = "Random seed")]
    pub(crate) seed: u64,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub(crate) device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub(crate) device_no: usize,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Produces:\n  \
                     {out}.feature_embedding.parquet  N × H per-feature embeddings, every node type\n  \
                     {out}.feature_types.parquet      the node type of every row\n  \
                     {out}.relations.parquet          one row per relation: types, weight, edge counts\n  \
                     {out}.log_likelihood.parquet     per-epoch train and eval loss\n  \
                     {out}.senna.json                 run manifest"
    )]
    pub(crate) out: Box<str>,
}

impl FneArgs {
    pub(crate) fn name_kind(&self) -> auxiliary_data::feature_names::FeatureNameKind {
        use auxiliary_data::feature_names::FeatureNameKind;
        if self.feature_name_exact {
            FeatureNameKind::Exact
        } else {
            FeatureNameKind::Gene {
                delim: self.feature_name_delim,
            }
        }
    }
}
