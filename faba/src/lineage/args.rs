//! `faba lineage` command-line surface.
//!
//! Split from [`super::run`] so the entry point reads as the sequence it is;
//! the semantics of each flag live in its own `long_help`.

use clap::{Args, ValueEnum};

/// 2D layout for plotting the trajectory.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum LayoutKind {
    /// No 2D layout.
    None,
    /// PHATE diffusion embedding (default) — the trajectory-appropriate layout
    /// that preserves branch/continuum structure (unlike UMAP/t-SNE), so it is
    /// on by default; pass `--layout none` to skip it.
    #[default]
    Phate,
    /// t-UMAP on a **cosine** kNN graph — sharper cluster separation than PHATE when
    /// the embedding is magnitude-heavy (PHATE's diffusion over-elongates those into
    /// convoluted arms). Cells, MST nodes, and curve points are embedded **jointly**
    /// in one fuzzy-kNN fit so they share the 2D space (no PHATE-style Nyström here).
    Umap,
}

/// Feature space the t-UMAP layout embeds on (`--layout umap`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum LayoutSpace {
    /// θ only — the identity manifold (current state).
    Identity,
    /// θ + δ — the NASCENT state (where each cell is heading). Splays the manifold
    /// toward the fates, so branches separate best on cosine t-UMAP. The default.
    #[default]
    Nascent,
    /// [θ | δ] concatenated — identity and velocity as separate cosine channels.
    Concat,
}

/// Whether the PHATE layout is warped along the confident velocity directions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum VelocityLayout {
    /// Warp only when enough edges are confidently oriented (default).
    #[default]
    Auto,
    /// Always warp along the selected directions.
    On,
    /// Never warp — the pure-θ PHATE manifold.
    Off,
}

// The advanced tuning knobs carry `hide_short_help` so `faba lineage -h` shows only the
// common flags; `--help` lists everything. `help_heading` buckets each flag into a category.
#[derive(Args, Debug)]
pub struct LineageArgs {
    #[arg(
        long,
        short = 'f',
        help_heading = "Input/output",
        help = "gem output prefix (reads {from}.cell_embedding.parquet and {from}.velocity.parquet)"
    )]
    pub from: Box<str>,

    #[arg(
        long,
        short = 'o',
        help_heading = "Input/output",
        help = "Output prefix (default: the gem prefix)"
    )]
    pub out: Option<Box<str>>,

    #[arg(
        long,
        help_heading = "Centroids & MST",
        help = "Number of MST node centroids K (default: min(cells / 10, 200))"
    )]
    pub n_centroids: Option<usize>,

    #[arg(
        long,
        default_value_t = 42,
        help_heading = "Centroids & MST",
        help = "RNG seed (reproducible centroids, edge directions, forest)"
    )]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = 100,
        hide_short_help = true,
        help_heading = "Centroids & MST",
        help = "k-means iterations for centroid initialization"
    )]
    pub kmeans_iter: usize,

    #[arg(
        long = "normalize-latent",
        hide_short_help = true,
        help_heading = "Centroids & MST",
        help = "L2-normalize θ (cosine geometry) for the fit AND layout \
                [default: raw θ / Euclidean]"
    )]
    pub normalize_latent: bool,

    #[arg(
        long = "no-edge-direction",
        help_heading = "Velocity direction & forest",
        help = "Skip the per-edge velocity direction test; forest = the geometric MST",
        long_help = "Skip the per-edge velocity direction test.\n\
            Every candidate edge is then geometry-only (abstained),\n\
            so the max-weight branching reduces to the geometric MST rooted by the hint chain —\n\
            the legacy behaviour with no velocity-informed cut/rewire."
    )]
    pub no_edge_direction: bool,

    #[arg(
        long = "no-orient-velocity",
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "Ignore velocity entirely (skip loading {from}.velocity.parquet)"
    )]
    pub no_orient_velocity: bool,

    #[arg(
        long,
        default_value_t = 4,
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "Nearest centroids added to the MST to form the directionality candidate set"
    )]
    pub edge_cand_knn: usize,

    #[arg(
        long,
        default_value_t = 200,
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "Cell bootstrap resamples for each edge's direction CI/SE"
    )]
    pub edge_direction_n_boot: usize,

    #[arg(
        long,
        default_value_t = 500,
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "Sign-flip permutation draws for each edge's direction p-value"
    )]
    pub edge_direction_n_perm: usize,

    #[arg(
        long,
        default_value_t = 0.05,
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "q cutoff and CI level (the abstain bar) for calling an edge's direction"
    )]
    pub edge_alpha: f64,

    #[arg(
        long,
        default_value_t = 2,
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "Minimum cells on an edge before its direction can be called (else abstain)"
    )]
    pub edge_min_cells: usize,

    #[arg(
        long,
        hide_short_help = true,
        help_heading = "Velocity direction & forest",
        help = "Forest granularity τ_root: virtual no-parent weight; higher ⇒ more trees. \
                Default = median selected arc weight."
    )]
    pub root_affinity: Option<f32>,

    #[arg(
        long = "root-type",
        help_heading = "Root selection",
        help = "Root the trajectory at the highest-confidence node of this cell type \
                (needs --markers), e.g. `--root-type HSC_MPP`. Marker-grounded and robust \
                to unreliable velocity; overrides --root-from-gem / velocity but is itself \
                overridden by --root-node / --root-cell."
    )]
    pub root_type: Option<Box<str>>,

    #[arg(
        long = "root-from-gem",
        help_heading = "Root selection",
        help = "Anchor the root at gem's velocity-DAG source: the modal MST node of the \
                low-τ region in {from}.dag_pseudotime.parquet. More robust than the per-edge \
                flux pick, while lineage still fits the curves. Overridden by \
                --root-node / --root-cell / --root-type; falls back to the flux root if the \
                file is absent or gem's DAG has no terminal structure (lineage_qc.json)."
    )]
    pub root_from_gem: bool,

    #[arg(
        long,
        hide_short_help = true,
        help_heading = "Root selection",
        help = "Force the root MST node by index (overrides velocity orientation)"
    )]
    pub root_node: Option<usize>,

    #[arg(
        long,
        hide_short_help = true,
        help_heading = "Root selection",
        help = "Force the root at the node nearest a named cell (overrides velocity)"
    )]
    pub root_cell: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 0.0,
        hide_short_help = true,
        help_heading = "Principal curves",
        help = "Gaussian kernel bandwidth in pseudotime units (0 = adaptive per curve)"
    )]
    pub curve_bandwidth: f32,

    #[arg(
        long,
        default_value_t = 100,
        hide_short_help = true,
        help_heading = "Principal curves",
        help = "Points sampled along each fitted principal curve"
    )]
    pub curve_resolution: usize,

    #[arg(
        long,
        default_value_t = 15,
        hide_short_help = true,
        help_heading = "Principal curves",
        help = "Max project-then-smooth iterations for the curves"
    )]
    pub max_iter: usize,

    #[arg(
        long,
        default_value_t = 1e-3,
        hide_short_help = true,
        help_heading = "Principal curves",
        help = "Convergence tolerance on mean |Δpseudotime| / range"
    )]
    pub tol: f32,

    #[arg(
        long,
        help_heading = "Marker annotation",
        help = "Marker TSV (gene<TAB>celltype) to name trajectory nodes by cell type",
        long_help = "Annotate each trajectory node with a cell type by term over-representation (the `faba annotate` core)\n\
                     run over the MST-node grouping, so the call carries the same permutation-calibrated confidence.\n\
                     pub(super) Input: a `gene<TAB>celltype` TSV (tab/comma/space delimited).\n\
                     Reads the co-embedded gene vectors from `{from}.feature_embedding.parquet` (spliced rows)\n\
                     and raw θ from `{from}.cell_embedding.parquet`.\n\
                     Writes `{out}.lineage_annot.*` (per-cell calls keyed by MST node)\n\
                     and `{out}.trajectory_annotation.parquet` (node → role[root|terminal|internal] → cell_type → confidence)."
    )]
    pub markers: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 500,
        hide_short_help = true,
        help_heading = "Marker annotation",
        help = "With --markers: permutation draws calibrating each node's over-representation"
    )]
    pub marker_num_perm: usize,

    #[arg(
        long,
        hide_short_help = true,
        help_heading = "Marker annotation",
        help = "Cell Ontology OBO file for the --markers ontology layer (needs --marker-label-cl)",
        long_help = "Optional. Adds a TreeBH Cell-Ontology layer over the per-node marker calls, as in `faba annotate`.\n\
                     Give the OBO graph here and the marker-type → CL id map via --marker-label-cl (both required together)."
    )]
    pub marker_obo: Option<Box<str>>,

    #[arg(
        long,
        hide_short_help = true,
        help_heading = "Marker annotation",
        help = "Curated `label<TAB>CL:id` TSV pairing marker types to CL ids (with --marker-obo)"
    )]
    pub marker_label_cl: Option<Box<str>>,

    #[arg(
        long = "no-bootstrap-markers",
        hide_short_help = true,
        help_heading = "Marker annotation",
        help = "[--markers] Turn OFF the stability bootstrap on the node calls",
        long_help = "Turn OFF the stability bootstrap on the node calls, naming each node by a bare point estimate.\n\n\
            The bootstrap is ON by default.\n\
            Each draw resamples every type's marker panel with replacement AND re-derives the k-means grouping;\n\
            the consensus is what ships, so a node's name carries the fraction of resamples that agreed on it.\n\n\
            This matters most for --root-type,\n\
            which picks the trajectory root as the highest-confidence node of a given type.\n\
            Without the bootstrap that `confidence` is a softmaxed test statistic rather than a reproducibility —\n\
            and the whole trajectory hangs off it.\n\n\
            Costs ~6 min at --marker-n-boot 200:\n\
            the replicate k-means has nothing to cache, unlike `faba annotate`'s kNN graph"
    )]
    pub no_bootstrap_markers: bool,

    #[arg(
        long,
        default_value_t = 200,
        hide_short_help = true,
        help_heading = "Marker annotation",
        help = "Bootstrap resamples on the node calls (--no-bootstrap-markers to disable)"
    )]
    pub marker_n_boot: usize,

    #[arg(
        long,
        default_value_t = 0.5,
        hide_short_help = true,
        help_heading = "Marker annotation",
        help = "[--markers] Minimum fraction of resamples the top label must win for \
                a node to be called at all (ignored under --no-bootstrap-markers)"
    )]
    pub marker_min_support: f32,

    #[arg(
        long,
        value_enum,
        default_value_t = LayoutKind::Phate,
        help_heading = "PHATE layout",
        help = "2D layout for plotting (default: phate). Emits {out}.{cells,nodes,curves}_2d.parquet; 'none' to skip"
    )]
    pub layout: LayoutKind,

    #[arg(
        long = "layout-space",
        value_enum,
        default_value_t = LayoutSpace::Nascent,
        help = "Feature space for --layout umap: identity (θ), nascent (θ+δ, default), or concat ([θ|δ])"
    )]
    pub layout_space: LayoutSpace,

    #[arg(
        long = "cluster-space",
        value_enum,
        default_value_t = LayoutSpace::Identity,
        help = "Feature space for the k-means grouping that drives annotation: identity (θ, default), \
                nascent (θ+δ), or concat ([θ|δ] — velocity separates committing progenitors θ alone cannot). \
                Marker scoring stays in θ regardless.",
        long_help = "Which cell features the annotation k-means groups on. `identity` (θ, the spliced \
                     state) is the default — cell TYPE is an identity question. `concat` ([θ|δ], each \
                     channel L2-normalised) additionally splits cells by their VELOCITY direction, so two \
                     transcriptionally-central cells heading to different fates land in different clusters — \
                     useful on a progenitor-enriched (e.g. CD34+) sample where θ alone can't resolve the \
                     committing structure. `nascent` (θ+δ) blends the two. The trajectory centroids and \
                     marker scoring are always recomputed in raw θ, so only the GROUPING changes."
    )]
    pub cluster_space: LayoutSpace,

    #[arg(
        long,
        value_enum,
        default_value_t = VelocityLayout::Auto,
        help_heading = "PHATE layout",
        help = "Warp the PHATE layout along confident velocity directions: auto (when \
                enough edges are oriented), on, or off"
    )]
    pub velocity_aware_layout: VelocityLayout,

    #[arg(
        long,
        default_value_t = 15,
        hide_short_help = true,
        help_heading = "PHATE layout",
        help = "PHATE kNN adaptive bandwidth (only with --layout phate)"
    )]
    pub phate_knn: usize,

    #[arg(
        long,
        default_value_t = 0,
        hide_short_help = true,
        help_heading = "PHATE layout",
        help = "PHATE diffusion time t (0 = auto-select at the von-Neumann-entropy knee)"
    )]
    pub phate_t: usize,

    #[arg(
        long,
        default_value_t = 2000,
        hide_short_help = true,
        help_heading = "PHATE layout",
        help = "PHATE landmark budget: above this many cells, PHATE runs on a \
                landmark subsample + Nyström lift (scales linearly). Raise it \
                if the layout looks thin/stringy on very large data."
    )]
    pub phate_landmarks: usize,
}
