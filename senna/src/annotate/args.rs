use clap::Args;

#[derive(Args, Debug)]
pub struct AnnotateArgs {
    #[arg(
        short = 'f',
        long = "from",
        required = true,
        help = "Run manifest produced by `senna topic|masked-topic|joint-topic|svd|joint-svd`"
    )]
    pub from: Box<str>,

    #[arg(
        short = 'c',
        long = "clusters",
        help = "Cluster parquet from `senna cluster` (cells × 1 cluster column,",
        long_help = "Cluster parquet from `senna cluster` (cells × 1 cluster column,\n\
                     `NaN` for unassigned). When omitted, the path is read from\n\
                     `manifest.cluster.clusters` if populated; otherwise annotate\n\
                     runs Leiden internally on the latent matrix from the manifest."
    )]
    pub clusters: Option<Box<str>>,

    #[arg(
        long = "knn",
        default_value_t = 15,
        help = "Nearest neighbors for the internal Leiden cosine-KNN graph",
        long_help = "Number of nearest neighbors for the cosine-KNN graph used by the\n\
                     internal Leiden clustering. Ignored when --clusters / manifest\n\
                     cluster path is provided."
    )]
    pub knn: usize,

    #[arg(
        long = "resolution",
        default_value_t = 1.0,
        help = "Modularity resolution (CPM) for the internal Leiden clustering",
        long_help = "Modularity resolution (CPM) for the internal Leiden clustering.\n\
                     Higher → more clusters. Ignored when clusters are supplied."
    )]
    pub resolution: f64,

    #[arg(
        long = "num-clusters",
        help = "Optional target cluster count for Leiden auto-resolution",
        long_help = "Optional target cluster count for Leiden auto-resolution. When set,\n\
                     resolution is binary-searched to approximate this. Ignored when\n\
                     clusters are supplied."
    )]
    pub num_clusters: Option<usize>,

    #[arg(
        long = "min-cluster-size",
        default_value_t = 2,
        help = "Minimum cluster size; smaller clusters become unassigned"
    )]
    pub min_cluster_size: usize,

    #[arg(
        long = "cluster-seed",
        help = "Seed for internal Leiden clustering (deterministic)"
    )]
    pub cluster_seed: Option<u64>,

    #[arg(
        short = 'm',
        long = "markers",
        default_value = "",
        help = "Marker-gene TSV: `gene<TAB>celltype` per line (one of --markers/--gaf/--gmt)",
        long_help = "Marker-gene TSV: `gene<TAB>celltype` per line. Flexible delimiter\n\
                     (tab / comma / space). Symbol / alias matching via `flexible_gene_match`.\n\
                     Exactly one gene-set source is required: --markers (curated cell-type\n\
                     markers), --gaf (GO annotations), or --gmt (MSigDB gene-sets)."
    )]
    pub markers: Box<str>,

    #[arg(
        long = "gaf",
        help = "GO annotation file (.gaf/.gaf.gz). Ontology mode: score each term as a \
                cross-cluster-contrasted module score on the cluster profile → per-cluster \
                signature ({out}.ontology_signature.tsv). --obo supplies term names."
    )]
    pub gaf: Option<Box<str>>,

    #[arg(
        long = "gmt",
        help = "MSigDB GMT gene-sets (`term<TAB>desc<TAB>genes…`). Ontology mode (as --gaf)"
    )]
    pub gmt: Option<Box<str>>,

    #[arg(
        long = "no-iea",
        default_value_t = false,
        help = "GAF only: drop IEA (electronic) annotations — the low-confidence bulk"
    )]
    pub no_iea: bool,

    #[arg(
        long = "min-gene-set",
        default_value_t = 15,
        help = "Ontology mode: minimum matched members for a term to be scored"
    )]
    pub min_gene_set: usize,

    #[arg(
        long = "max-gene-set",
        default_value_t = 500,
        help = "Ontology mode: maximum matched members (size window; excludes \
                near-universal terms)"
    )]
    pub max_gene_set: usize,

    #[arg(
        short = 'o',
        long = "out",
        help = "Output prefix for annotation artifacts",
        long_help = "Output prefix for annotation artifacts. When omitted, derived\n\
                     from `--from` by stripping `.senna.json` (or `.json`) — e.g.\n\
                     `--from temp.senna.json` → `--out temp`."
    )]
    pub out: Option<Box<str>>,

    #[arg(short = 'v', long, help = "Verbose logging")]
    pub verbose: bool,

    #[arg(
        long = "block-size",
        default_value_t = 1024,
        help = "Cells per CSC read block when streaming raw counts for per-cluster aggregation",
        long_help = "Cells per CSC read block when streaming raw counts for per-cluster\n\
                     aggregation and NB-Fisher trend fitting. Larger blocks → fewer reads\n\
                     but more memory."
    )]
    pub block_size: usize,

    #[arg(
        long = "num-draws",
        default_value_t = 1000,
        help = "Random gene-set draws per cell type (Efron–Tibshirani moments)",
        long_help = "Number of random gene-set draws per celltype for the\n\
                     Efron–Tibshirani row-randomization moments (used to restandardize\n\
                     both observed and permuted ES)."
    )]
    pub num_draws: usize,

    #[arg(
        long = "num-perm",
        default_value_t = 500,
        help = "Number of PB-level sample permutations for the correlation-preserving null",
        long_help = "Number of PB-level sample permutations for the correlation-preserving\n\
                     null (shuffle pb_membership, recompute β̃ = pb_gene · shuffled,\n\
                     ES per permutation, pool across clusters). Set 0 to fall back to\n\
                     row-randomization-based p-values (useful for small nClusters)."
    )]
    pub num_perm: usize,

    #[arg(
        long = "min-markers",
        default_value_t = 3,
        help = "Drop a cell type with fewer than this many matched markers",
        long_help = "Minimum matched markers before a cell type is allowed to compete.\n\n\
            A type below this is not weakly supported, it is UNSUPPORTED: an enrichment\n\
            walk over one or two genes is noise, and the winner's curse then hands the\n\
            cluster to whichever noisy panel happened to spike.\n\n\
            A dropped type keeps its column in every output. It simply never wins a\n\
            cluster.\n\n\
            Floored at 2: you cannot resample a single point"
    )]
    pub min_markers: usize,

    #[arg(
        long = "fdr-alpha",
        default_value_t = 0.10,
        help = "FDR α for the Q-matrix threshold"
    )]
    pub fdr_alpha: f32,

    #[arg(
        long = "q-temperature",
        default_value_t = 1.0,
        help = "Softmax temperature used when row-normalizing Q over significant entries",
        long_help = "Softmax temperature used when row-normalizing Q over significant\n\
                     entries. Lower → sharper; higher → more uniform."
    )]
    pub q_temperature: f32,

    #[arg(
        long = "min-confidence",
        default_value_t = 0.0,
        help = "Minimum cell-level confidence to emit a concrete label",
        long_help = "Minimum cell-level confidence to emit a concrete label; below this\n\
                     cells are labeled `unassigned` in the argmax TSV."
    )]
    pub min_confidence: f32,

    #[arg(
        long = "seed",
        default_value_t = 42,
        help = "RNG seed (deterministic; affects row randomization)"
    )]
    pub seed: u64,

    #[arg(
        long = "no-clean",
        help = "Keep existing {out}.* annotation outputs (default: erase the explicit annotation set first — never the embedding/manifest — for a fresh re-run)"
    )]
    pub no_clean: bool,

    #[arg(
        long = "preload-data",
        default_value_t = false,
        help = "Preload columns into memory after opening the zarr/h5 backend",
        long_help = "Preload columns into memory after opening the zarr/h5 backend.\n\
                     On slow disks this trades memory for I/O latency on later block reads."
    )]
    pub preload_data: bool,

    #[arg(
        long = "no-empirical-specificity",
        default_value_t = false,
        help = "Disable data-aware specificity re-weighting of marker genes",
        long_help = "Disable data-aware specificity re-weighting of marker genes. By\n\
                     default, each marker is multiplied by an empirical specificity\n\
                     score derived from the cluster expression matrix (`max simplex\n\
                     value across clusters`, rescaled to [0, 1]) — this suppresses\n\
                     markers that fire broadly (e.g. GZMB shared between NK and CD8\n\
                     effector). Set this flag to fall back to IDF-only weighting."
    )]
    pub no_empirical_specificity: bool,

    //////////////////////////////////////////////////
    // optional inline ontology annotation (TreeBH) //
    //////////////////////////////////////////////////
    #[arg(
        long = "obo",
        help = "Cell Ontology .obo (e.g. cl-basic.obo). Given WITH --label-cl, runs TreeBH \
                ontology calling inline → {out}.ontology_assignment.tsv + .ontology_node_mass.parquet"
    )]
    pub obo: Option<Box<str>>,

    #[arg(
        long = "label-cl",
        help = "Curated `label<TAB>CL:id` map, one row per marker celltype. Required together with --obo"
    )]
    pub label_cl: Option<Box<str>>,

    #[arg(
        long = "ontology-fdr-q",
        default_value_t = 0.1,
        help = "Ontology TreeBH per-level FDR target (lower → descends less, abstains more)"
    )]
    pub ontology_fdr_q: f64,

    #[arg(
        long = "ontology-by",
        default_value_t = false,
        help = "Ontology: Benjamini–Yekutieli within families (any dependence; more conservative)"
    )]
    pub ontology_by: bool,

    #[arg(
        long = "no-gene-strata",
        help = "Draw the null gene sets uniformly instead of within gene-abundance strata",
        long_help = "Draw the null gene sets uniformly over all genes instead of within\n\
            gene-abundance strata (GOseq). NOT recommended — this restores a known bias.\n\n\
            The enrichment score is standardized against random gene sets, and that\n\
            standardization is the decision variable (the label is the argmax over\n\
            celltypes of it). A uniform draw is ~30% undetected genes, which sort to the\n\
            bottom of every ranking and can never be enriched — so it is trivially easy to\n\
            beat, and easy to beat by an amount that DIFFERS PER CELLTYPE, because panels\n\
            differ in how well-expressed their markers are.\n\n\
            Measured on BMMNC with the uniform null: a celltype's mean es_std was a\n\
            perfectly monotone function of its markers' mean expression (Spearman +1.000\n\
            across all 8 types). Stratifying puts the abundance advantage on both sides of\n\
            the comparison, where it cancels.\n\n\
            This is GOseq's gene-length correction [Young et al. 2010] in expression space.\n\
            Kept only as an escape hatch and to reproduce pre-0.4 outputs"
    )]
    pub no_gene_strata: bool,

    ////////////////////////////////////
    // marker-panel stability bootstrap //
    ////////////////////////////////////
    #[arg(
        long = "no-bootstrap-markers",
        help = "Turn OFF the stability bootstrap and ship a bare point estimate",
        long_help = "Turn OFF the stability bootstrap and ship a bare point estimate.\n\n\
            The bootstrap is ON by default. Each draw resamples every celltype's marker\n\
            panel with replacement, re-walks the enrichment score and re-calls the FDR;\n\
            the consensus is what ships. So every call carries the fraction of resamples\n\
            that agreed on it, and a call that cannot hold up across them abstains.\n\n\
            NOTE the support here is PER-CLUSTER, not per-cell: on this path a cell's\n\
            label IS its cluster's label (the cell→cluster membership is one-hot), so\n\
            there is no per-cell decision to resample. It is written out as\n\
            `cluster_label_support`. Unlike `annotate-by-projection`, this bootstrap does\n\
            NOT re-derive the clustering — that would mean re-streaming the raw counts off\n\
            the backend once per draw — so it sees the variance the PANEL contributes and\n\
            not the variance the PARTITION contributes, and is optimistic accordingly"
    )]
    pub no_bootstrap_markers: bool,

    #[arg(
        long = "n-boot",
        default_value_t = 200,
        help = "Bootstrap resamples (0 or --no-bootstrap-markers to disable)"
    )]
    pub n_boot: usize,

    #[arg(
        long = "boot-num-draws",
        default_value_t = 100,
        help = "Random gene sets per bootstrap draw for the restandardization moments",
        long_help = "Random gene sets drawn per bootstrap draw, per celltype, for the\n\
            restandardization moments.\n\n\
            This is the cost centre of the bootstrap (n_boot x C x this x K enrichment\n\
            walks). It cannot be replaced by the observed row-randomization null: that one\n\
            uses binary weights at the panel's nominal size, while a resampled panel has\n\
            ~0.632x the distinct genes AND a dispersed weight multiset. Standardizing a\n\
            resampled score against an unmatched null inflates small panels — which is\n\
            precisely the winner's curse the bootstrap exists to remove.\n\n\
            The moments only need ~10% relative accuracy on the SD, so 100 is plenty;\n\
            lower it first if the bootstrap is too slow"
    )]
    pub boot_num_draws: usize,

    #[arg(
        long = "min-support",
        default_value_t = 0.5,
        help = "Minimum fraction of resamples the top label must win for a cluster to be called",
        long_help = "Minimum fraction of resamples the top label must win for the cluster to\n\
            be called at all.\n\n\
            NOTE this bar is NOT scale-free. With C celltypes, chance agreement is 1/C, so\n\
            0.5 sits at ~3x chance on a 6-type panel and ~12x chance on a 24-type one —\n\
            the same value is a different test on different panels, and their abstention\n\
            rates are not comparable. --abstain-separable (a sign test) avoids that"
    )]
    pub min_support: f32,

    #[arg(
        long = "abstain-separable",
        conflicts_with = "min_support",
        help = "Abstain by a sign test instead of the --min-support threshold",
        long_help = "Abstain by a TEST rather than a threshold.\n\n\
            Keep the top label only if it beat the runner-up by more than resampling\n\
            noise — an exact binomial sign test at --abstain-alpha. No magic number, and\n\
            unlike --min-support it means the same thing whatever the number of celltypes"
    )]
    pub abstain_separable: bool,

    #[arg(
        long = "abstain-alpha",
        default_value_t = 0.05,
        help = "[--abstain-separable] Significance level for the top-vs-runner-up sign test"
    )]
    pub abstain_alpha: f64,

    #[arg(
        long = "set-coverage",
        default_value_t = 0.8,
        help = "Coverage of the reported `label_set` (the mixed annotation)",
        long_help = "Coverage of the reported `label_set` — the smallest set of labels\n\
            accounting for this share of the resamples.\n\n\
            A cluster that cannot be given ONE label can still be given two, and\n\
            `HSPC/LMPP` is a far better answer than `unassigned`"
    )]
    pub set_coverage: f32,

    #[arg(
        long = "max-set-size",
        default_value_t = 3,
        help = "Largest `label_set` worth printing (a 4-way tie is not an annotation)"
    )]
    pub max_set_size: usize,
}

/// `senna annotate-by-projection` — firm marker-set annotation by projection
/// onto a co-embedded feature space (bge / fne / resolve-embedding-space).
/// Embedding-grounded (no raw-count re-read), complementary to
/// `annotate-by-enrichment`. Drives the shared firm term-ORA core.
#[derive(Args, Debug)]
pub struct AnnotateProjectionArgs {
    #[arg(
        short = 'f',
        long = "from",
        required = true,
        help = "Run manifest from a co-embedding run (`senna bge|fne|resolve-embedding-space`)",
        long_help = "Run manifest with a co-embedded gene space — `senna bge`, `fne`, or\n\
                     `resolve-embedding-space` (reads `outputs.feature_embedding` +\n\
                     `outputs.cell_embedding`, falling back to `outputs.latent` for the\n\
                     cell side on plain bge/fne). topic/svd runs have no genes-on-the-\n\
                     cell-manifold embedding — use `annotate-by-enrichment` for those."
    )]
    pub from: Box<str>,

    #[arg(
        short = 'm',
        long = "markers",
        required = true,
        help = "Marker-gene TSV: `gene<TAB>celltype` per line (tab/comma/space delimited)"
    )]
    pub markers: Box<str>,

    #[arg(
        short = 'o',
        long = "out",
        help = "Output prefix (default: `--from` with `.senna.json`/`.json` stripped)"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long = "knn",
        default_value_t = 30,
        help = "k for the cosine cell kNN graph fed to Leiden clustering"
    )]
    pub knn: usize,

    #[arg(
        long = "resolution",
        default_value_t = 1.0,
        help = "Leiden resolution for cell clustering (higher → more, finer clusters)"
    )]
    pub resolution: f64,

    #[arg(
        long = "num-perm",
        default_value_t = 500,
        help = "Permutation draws calibrating the over-representation null (0 = analytic p only)"
    )]
    pub num_perm: usize,

    #[arg(
        long = "seed",
        default_value_t = 42,
        help = "RNG seed (clustering + permutation null)"
    )]
    pub seed: u64,

    #[arg(
        long = "min-markers",
        default_value_t = 3,
        help = "Drop a cell type with fewer than this many usable markers",
        long_help = "Minimum usable markers before a cell type is allowed to compete.\n\n\
            A type below this is not weakly located, it is UNLOCATED. The mean of one or\n\
            two points has no direction worth the name, and a centroid built from too few\n\
            markers lands short — near the middle of the cell cloud, where it is close to\n\
            EVERY cell at once. It does not compete weakly; it becomes a magnet and takes\n\
            the dataset.\n\n\
            A dropped type keeps its column in every output. It simply never wins a cell.\n\n\
            Floored at 2: you cannot resample a single point"
    )]
    pub min_markers: usize,

    #[arg(
        long = "no-idf",
        help = "Disable IDF down-weighting of markers shared across many types"
    )]
    pub no_idf: bool,

    #[arg(
        long = "no-assign-qc",
        help = "Keep every cell→term assignment (skip the distance-outlier prune)"
    )]
    pub no_assign_qc: bool,

    #[arg(
        long = "assign-mad",
        default_value_t = 2.5,
        help = "Outlier gate: prune a cell whose distance to its centroid exceeds median + k·MAD"
    )]
    pub assign_mad: f64,

    #[arg(
        long = "fdr-alpha",
        default_value_t = 0.1,
        help = "FDR α for the per-cluster term call + Q sparsity (BH on the permutation p)"
    )]
    pub fdr_alpha: f32,

    #[arg(
        long = "q-temperature",
        default_value_t = 1.0,
        help = "Softmax temperature when row-normalizing Q over significant terms"
    )]
    pub q_temperature: f32,

    #[arg(
        long = "obo",
        help = "Cell Ontology .obo (e.g. cl-basic.obo). With --label-cl, runs TreeBH ontology \
                calling on the cluster × term matrix → {out}.ontology_assignment.tsv"
    )]
    pub obo: Option<Box<str>>,

    #[arg(
        long = "label-cl",
        help = "Curated `label<TAB>CL:id` map, one row per marker celltype. Required with --obo"
    )]
    pub label_cl: Option<Box<str>>,

    #[arg(
        long = "ontology-fdr-q",
        default_value_t = 0.1,
        help = "Ontology TreeBH per-level FDR target (lower → descends less, abstains more)"
    )]
    pub ontology_fdr_q: f64,

    #[arg(
        long = "ontology-by",
        help = "Ontology: Benjamini–Yekutieli within families (any dependence; more conservative)"
    )]
    pub ontology_by: bool,

    #[arg(
        long = "panel-perm",
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
        long = "support-perm",
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
        long = "n-boot",
        default_value_t = 200,
        help = "Bootstrap resamples (0 or --no-bootstrap-markers to disable)"
    )]
    pub n_boot: usize,

    #[arg(
        long = "no-recluster",
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
        long = "min-support",
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
        long = "abstain-separable",
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
        long = "abstain-alpha",
        default_value_t = 0.05,
        help = "[--abstain-separable] Significance level for the top-vs-runner-up sign test"
    )]
    pub abstain_alpha: f64,

    #[arg(
        long = "set-coverage",
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
        long = "max-set-size",
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
        long = "no-clean",
        help = "Keep existing {out}.* projection outputs (default: erase the explicit set first)"
    )]
    pub no_clean: bool,

    #[arg(short = 'v', long, help = "Verbose logging")]
    pub verbose: bool,
}

/// `senna annotate-ontology` — hierarchical multi-resolution cell-type calling
/// (TreeBH) on the Cell Ontology, post-processing an `annotate-by-enrichment`
/// run's cluster × celltype matrix.
#[derive(Args, Debug)]
pub struct AnnotateOntologyArgs {
    #[arg(
        short = 'f',
        long = "from",
        required = true,
        help = "Run manifest already annotated by `senna annotate-by-enrichment`",
        long_help = "Run manifest already annotated by `senna annotate-by-enrichment`\n\
                     (reads `annotate.cluster_celltype_q` and its sibling\n\
                     `*_es_std` / `*_p` matrices)."
    )]
    pub from: Box<str>,

    #[arg(
        long = "label-cl",
        required = true,
        help = "Curated `label<TAB>CL:id` TSV mapping each marker celltype to a Cell Ontology term"
    )]
    pub label_cl: Box<str>,

    #[arg(
        long = "obo",
        required = true,
        help = "Cell Ontology OBO file (e.g. cl-basic.obo)",
        long_help = "Cell Ontology OBO file. Fetch the basic release with:\n\
                     curl -sSL https://github.com/obophenotype/cell-ontology/\
                     releases/latest/download/cl-basic.obo -o cl-basic.obo"
    )]
    pub obo: Box<str>,

    #[arg(
        short = 'o',
        long = "out",
        help = "Output prefix (defaults to `--from` with `.senna.json`/`.json` stripped)"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long = "fdr-q",
        default_value_t = 0.1,
        help = "Per-level selective-FDR target (TreeBH); lower → descends less, abstains more"
    )]
    pub fdr_q: f64,

    #[arg(
        long = "by",
        default_value_t = false,
        help = "Benjamini–Yekutieli within families (valid under any dependence; more conservative)"
    )]
    pub by: bool,

    #[arg(
        long = "use-perm-p",
        default_value_t = false,
        help = "Force the (saturated) permutation p-values instead of the default z→p",
        long_help = "Force `cluster_celltype_p`. By default the walk scores on Φ(−z) using\n\
                     the correlation-preserving permutation z (`*_perm_z`) when present,\n\
                     else the row-randomization restandardized ES (`*_es_std`). The\n\
                     permutation p is resolution-limited (≈1/B) and rarely preferable."
    )]
    pub use_perm_p: bool,

    #[arg(short = 'v', long, help = "Verbose logging")]
    pub verbose: bool,
}
