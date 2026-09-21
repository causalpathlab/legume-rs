use crate::common::*;
use crate::data::cell_membership::CellMembership;
use crate::data::conversion::*;
use crate::data::dna::Dna;
use crate::data::dna_stat_map::*;
use crate::data::util_htslib::*;
use crate::editing::sifter::*;
use crate::editing::ConversionSite;
use crate::quant::*;

use dashmap::DashMap as HashMap;
use dashmap::DashSet as HashSet;
use genomic_data::gff::{GeneId, GffRecordMap};
use rust_htslib::faidx;
use std::sync::{Arc, Mutex};

/// Padding around target region when reading BAM files
const BAM_READ_PADDING: i64 = 1;

/// How per-observation weights are computed for the per-gene mixture model.
///
/// Both modes feed the same value to the EM and to the output sparse matrix
/// (matrix entries are fractional under `Posterior`). The CLI default is
/// `Posterior`; `Converted` preserves the pre-2026 behavior bit-for-bit.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Default, serde::Serialize)]
#[clap(rename_all = "lowercase")]
pub enum MixtureWeightMode {
    /// Weight each observation by raw converted-read count `c_i` (legacy).
    Converted,
    /// Weight by Beta-posterior regularized effective count
    /// `w_i = n_i · (c_i + α) / (n_i + α + β)`, where `n_i = c_i + u_i`.
    #[default]
    Posterior,
}

/// Unified parameters for base conversion (m6A and A-to-I) discovery and quantification.
pub struct ConversionParams {
    pub genome_file: Box<str>,
    pub wt_bam_files: Vec<Box<str>>,
    pub gene_barcode_tag: Box<str>,
    pub cell_barcode_tag: Box<str>,
    pub include_missing_barcode: bool,
    /// Candidacy floor on total coverage at a site (signal + control for m6A,
    /// ref + alt for A-to-I). The producer applies no other threshold: every
    /// site that clears this and `min_conversion` is written and quantified.
    pub min_coverage: usize,
    pub min_conversion: usize,
    /// Sequencing-error rate ε for the beta-binomial editing null.
    pub error_rate: f64,
    /// Overdispersion ρ for the beta-binomial editing null (0 ⇒ binomial).
    pub overdispersion: f64,
    pub backend: SparseIoBackend,
    /// Wrap zarr output in a `.zarr.zip` archive (no effect for HDF5).
    pub zip: bool,
    pub output: Box<str>,
    pub cell_membership_file: Option<Box<str>>,
    pub membership_barcode_col: usize,
    pub membership_celltype_col: usize,
    pub exact_barcode_match: bool,
    pub mod_type: ModificationType,
    pub min_base_quality: u8,
    pub min_mapping_quality: u8,
    /// Per-observation weighting used by `run_mixture_model`.
    pub mixture_weight_mode: MixtureWeightMode,
    /// Beta(α, β) prior parameter α for `MixtureWeightMode::Posterior`.
    pub mixture_prior_alpha: f32,
    /// Beta(α, β) prior parameter β for `MixtureWeightMode::Posterior`.
    pub mixture_prior_beta: f32,
    /// UMI BAM tag for read deduplication. `None` disables UMI dedup so
    /// every aligned read at a position contributes to the base counts.
    pub umi_tag: Option<Box<str>>,
    /// MUT (catalytically-dead YTHmut) control BAMs for the m6A WT-vs-MUT
    /// contrast. Pooled into one background. Empty for A-to-I (single-sample).
    pub mut_bam_files: Vec<Box<str>>,
    /// Editing-competent cells from [`crate::editing::cell_activity`], restricting
    /// the **signal arm only**, keyed by signal BAM path.
    ///
    /// Per BAM, not one flat set, for the same reason `GeneCountQc::cells_by_batch`
    /// is: barcodes are only unique within a single library, so one shared set
    /// would let a competent cell in library A rescue the null cell that happens
    /// to share its barcode in library B. `None`, or a BAM absent from the map,
    /// keeps every cell for that library.
    ///
    /// Applied at discovery, so each site's WT counts are de-diluted as they are
    /// first computed and nothing downstream has to be recomputed.
    ///
    /// Never applied to `mut_bam_files`: selecting control cells on apparent
    /// activity selects on background, which inflates the null and biases the
    /// contrast toward missing real sites.
    pub competent_cells: Option<Arc<crate::editing::cell_activity::CompetentCells>>,
}

/// Defaults for the single-sample (A-to-I) beta-binomial sequencing-error
/// null: error rate ε and overdispersion ρ. m6A does not use them (its test is
/// the WT-vs-MUT contrast), so `faba dartseq` exposes no flag for them.
pub const DEFAULT_EDIT_ERROR_RATE: f64 = 0.01;
/// See [`DEFAULT_EDIT_ERROR_RATE`].
pub const DEFAULT_EDIT_OVERDISPERSION: f64 = 0.1;

/// Defaults for the m6A candidacy floors, shared by `faba dartseq`'s
/// `--min-coverage` / `--min-conversion` and `faba all`'s `--m6a-min-coverage` /
/// `--m6a-min-conversion`.
///
/// Defined once because the two MUST agree: they drifted before, and `faba all`
/// called m6A sites a hand-run `faba dartseq` rejected. A shared const makes
/// agreement a compile-time fact.
///
/// The producer is inclusive by design. `min_coverage` is on TOTAL coverage
/// (signal + control) and `min_conversion` on converted signal reads; at 1 / 1
/// the only site excluded is one with no converted read at all, i.e. an empty
/// row. Every other decision (p-value, odds ratio, control depth, cells per
/// site) is a column in the sites parquet and a flag on `faba qc`, so it can be
/// revisited without rescanning BAMs.
pub const DEFAULT_M6A_MIN_COVERAGE: usize = 1;
/// See [`DEFAULT_M6A_MIN_COVERAGE`].
pub const DEFAULT_M6A_MIN_CONVERSION: usize = 1;

/// Defaults for the A-to-I candidacy floors, shared by `faba atoi` and
/// `faba all --atoi-*`. A-to-I has no motif anchor, so at 1 / 1 every reference
/// A with a single mismatching read would be a site; these floors bound the
/// candidate set. `faba dartseq --detect-atoi` used to carry its own copy at
/// 10 / 5, so the same BAM produced a different A-to-I site list depending on
/// which command ran it.
pub const DEFAULT_ATOI_MIN_COVERAGE: usize = 5;
/// See [`DEFAULT_ATOI_MIN_COVERAGE`].
pub const DEFAULT_ATOI_MIN_CONVERSION: usize = 3;

impl ConversionParams {
    /// If a UMI tag is configured, enable UMI dedup on the given freq map.
    fn apply_umi(&self, map: &mut crate::data::dna_stat_map::DnaBaseFreqMap<'_>) {
        if let Some(ref tag) = self.umi_tag {
            map.set_umi_tag(tag, &self.cell_barcode_tag);
        }
    }

    /// Restrict a **signal-arm** map to `bam_file`'s editing-competent cells, if
    /// any were called for it. Must be re-applied per BAM inside the read loop —
    /// the sets are per library. Deliberately never applied to a control map —
    /// see [`Self::competent_cells`].
    fn restrict_to_competent_cells(
        &self,
        map: &mut crate::data::dna_stat_map::DnaBaseFreqMap<'_>,
        bam_file: &str,
    ) {
        map.set_keep_cells(
            &self.cell_barcode_tag,
            self.competent_cells
                .as_ref()
                .and_then(|by_batch| by_batch.get(bam_file))
                .cloned(),
        );
    }

    /// BAMs that receive per-cell quantification in the second pass: the signal
    /// (wt) samples AND the control (mut) samples. The mut matrices are a sanity
    /// check (their m6A should read ~background) and still feed gem. For A-to-I
    /// `mut_bam_files` is empty, so this is just the wt set.
    pub fn quant_bam_files(&self) -> Vec<Box<str>> {
        let (files, dropped) = unique_bam_files(
            self.wt_bam_files
                .iter()
                .chain(self.mut_bam_files.iter())
                .cloned(),
        );
        if dropped > 0 {
            log::warn!(
                "{dropped} BAM file(s) listed in both the signal and control \
                 (--control-bam) sets; quantifying each once to avoid double counting"
            );
        }
        files
    }

    /// Create a ConversionSifter with these parameters
    pub fn create_sifter<'a>(
        &self,
        faidx: &'a faidx::Reader,
        chr: &'a str,
        capacity: usize,
    ) -> ConversionSifter<'a> {
        ConversionSifter {
            faidx,
            chr,
            min_coverage: self.min_coverage,
            min_conversion: self.min_conversion,
            error_rate: self.error_rate,
            overdispersion: self.overdispersion,
            mod_type: self.mod_type.clone(),
            candidate_sites: Vec::with_capacity(capacity),
        }
    }

    /// Resolve the user-facing target path (`.zarr.zip` when applicable) and
    /// the underlying write path. After writing the backend, call
    /// [`BackendOutputPath::finalize`] to zip the staging directory.
    pub fn backend_output_path(&self, batch_name: &str) -> crate::quant::BackendOutputPath {
        crate::quant::BackendOutputPath::new(&self.output, batch_name, &self.backend, self.zip)
    }

    /// Minimum number of positions required to attempt site discovery.
    /// m6A requires a triplet (3 consecutive positions), A-to-I needs only 1.
    fn min_length_for_testing(&self) -> usize {
        match self.mod_type {
            ModificationType::M6A { .. } => 3,
            ModificationType::AtoI => 1,
        }
    }

    /// Load cell membership from file if configured
    pub fn load_membership(&self) -> anyhow::Result<Option<CellMembership>> {
        if let Some(ref path) = self.cell_membership_file {
            let m = CellMembership::from_file(
                path,
                self.membership_barcode_col,
                self.membership_celltype_col,
                !self.exact_barcode_match,
            )?;
            info!(
                "Loaded {} cell barcodes from membership file: {}",
                m.num_cells(),
                path
            );
            info!("Prefix matching: {}", !self.exact_barcode_match);
            Ok(Some(m))
        } else {
            Ok(None)
        }
    }
}

////////////////////////////////
// FIRST PASS: Site discovery //
////////////////////////////////

/// Find all putative conversion sites across the genome, keyed by gene and
/// sorted (genes by id, sites by position) so every writer downstream is
/// reproducible without sorting again.
///
/// Discovery is the whole producer-side decision. A site is materialized when
/// it clears the sifter's coverage / minimum-conversion floors and nothing
/// else: its p-value, odds ratio and control depth are carried on the site and
/// written to the sites parquet, and every site is quantified in the second
/// pass. Thresholding them is `faba qc`'s job. The unit is always the site;
/// pooling a gene's putative sites into one test was removed because a
/// gene-level verdict cannot say WHICH C carries the mark.
///
/// For m6A with cell_membership: uses per-cell-type discovery.
/// For m6A without membership or for AtoI: uses bulk statistics.
pub fn find_all_conversion_sites(
    gff_map: &GffRecordMap,
    params: &ConversionParams,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<HashMap<GeneId, Vec<ConversionSite>>> {
    let njobs = gff_map.len();
    info!(
        "Searching {} conversion sites over {} blocks",
        params.mod_type.label(),
        njobs
    );

    // Validate reference genome
    info!("Loading reference genome: {}", params.genome_file);
    load_fasta_index(&params.genome_file)?;

    let arc_gene_sites = Arc::new(HashMap::<GeneId, Vec<ConversionSite>>::default());

    gff_map
        .records()
        .par_iter()
        .progress_with(new_progress_bar(njobs as u64))
        .try_for_each_init(
            || {
                let faidx = load_fasta_index(&params.genome_file)
                    .expect("fasta index validated upfront; load per worker thread");
                let cache = crate::data::bam_io::BamReaderCache::new();
                (faidx, cache)
            },
            |(faidx, cache), rec| -> anyhow::Result<()> {
                find_sites_in_gene(
                    rec,
                    params,
                    faidx,
                    cache,
                    arc_gene_sites.clone(),
                    cell_membership,
                )
            },
        )?;

    let gene_sites = Arc::try_unwrap(arc_gene_sites)
        .map_err(|_| anyhow::anyhow!("failed to release gene_sites"))?;

    let sites = sort_sites(gene_sites);
    let n_sites: usize = sites.iter().map(|e| e.value().len()).sum();
    info!(
        "{}: {} putative sites over {} genes (no producer-side test; \
         threshold pv / log_odds / coverage with `faba qc`)",
        params.mod_type.label(),
        n_sites,
        sites.len(),
    );
    Ok(sites)
}

/// Fix the order once, here, not at each writer. `gene_sites` is a DashMap, so
/// its iteration order is whatever the parallel scan's sharding produced, and
/// the site parquet would inherit it. Ordering the discovery result makes every
/// consumer reproducible instead of asking each writer to remember to sort.
fn sort_sites(
    gene_sites: HashMap<GeneId, Vec<ConversionSite>>,
) -> HashMap<GeneId, Vec<ConversionSite>> {
    let mut genes: Vec<(GeneId, Vec<ConversionSite>)> = gene_sites.into_iter().collect();
    genes.sort_by(|a, b| a.0.cmp(&b.0));
    let out = HashMap::<GeneId, Vec<ConversionSite>>::default();
    for (gid, mut sites) in genes {
        sites.sort_by_key(|s| s.primary_pos());
        out.insert(gid, sites);
    }
    out
}

/// Per-gene site discovery: reads WT and MUT BAM files, creates
/// sifter, dispatches via scan().
///
/// `faidx_reader` is owned by the calling rayon worker (one per thread, reused
/// across all genes scheduled to that worker) — see `find_conversion_sites`.
fn find_sites_in_gene(
    gff_record: &GffRecord,
    params: &ConversionParams,
    faidx_reader: &faidx::Reader,
    cache: &mut crate::data::bam_io::BamReaderCache,
    arc_gene_sites: Arc<HashMap<GeneId, Vec<ConversionSite>>>,
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<()> {
    let gene_id = gff_record.gene_id.clone();
    let strand = &gff_record.strand;
    let chr = gff_record.seqname.as_ref();

    // Per-cell-group discovery whenever a membership is supplied (mass
    // enrichment); otherwise bulk. Applies to both m6A (WT-vs-MUT contrast) and
    // A-to-I (control-free, reference-anchored) — the stratified helper builds a
    // pooled control only when `mut_bam_files` is non-empty, so A-to-I routes to
    // the reference-anchored beta-binomial per stratum.
    let candidate_sites = match cell_membership {
        Some(membership) => find_sites_with_celltype_stats(
            gff_record,
            params,
            faidx_reader,
            cache,
            chr,
            strand,
            membership,
        )?,
        None => find_sites_with_bulk_stats(gff_record, params, faidx_reader, cache, chr, strand)?,
    };

    if !candidate_sites.is_empty() {
        arc_gene_sites.insert(gene_id, candidate_sites);
    }

    Ok(())
}

/// Find conversion sites using bulk/marginal statistics (no cell type info).
fn find_sites_with_bulk_stats(
    gff_record: &GffRecord,
    params: &ConversionParams,
    faidx_reader: &faidx::Reader,
    cache: &mut crate::data::bam_io::BamReaderCache,
    chr: &str,
    strand: &Strand,
) -> anyhow::Result<Vec<ConversionSite>> {
    let mut wt_base_freq_map = DnaBaseFreqMap::new();
    wt_base_freq_map.set_quality_thresholds(params.min_base_quality, params.min_mapping_quality);
    params.apply_umi(&mut wt_base_freq_map);

    for wt_file in &params.wt_bam_files {
        params.restrict_to_competent_cells(&mut wt_base_freq_map, wt_file);
        wt_base_freq_map.update_from_gene_cached(
            cache,
            wt_file,
            gff_record,
            &params.gene_barcode_tag,
            params.include_missing_barcode,
        )?;
    }

    let positions = wt_base_freq_map.sorted_positions();

    if positions.len() < params.min_length_for_testing() {
        return Ok(Vec::new());
    }

    let mut sifter = params.create_sifter(faidx_reader, chr, positions.len());

    let wt_freq = wt_base_freq_map
        .marginal_frequency_map()
        .ok_or_else(|| anyhow::anyhow!("failed to count wt freq"))?;

    // Pooled MUT (control) frequencies for the m6A WT-vs-MUT contrast. Empty for
    // A-to-I (single-sample) → skip piling the control BAMs and pass `None`,
    // which routes the sifter to its reference-anchored beta-binomial path.
    let mut mut_base_freq_map = DnaBaseFreqMap::new();
    let mut_freq = if params.mut_bam_files.is_empty() {
        None
    } else {
        mut_base_freq_map
            .set_quality_thresholds(params.min_base_quality, params.min_mapping_quality);
        params.apply_umi(&mut mut_base_freq_map);
        for mut_file in &params.mut_bam_files {
            mut_base_freq_map.update_from_gene_cached(
                cache,
                mut_file,
                gff_record,
                &params.gene_barcode_tag,
                params.include_missing_barcode,
            )?;
        }
        mut_base_freq_map.marginal_frequency_map()
    };

    let forward = matches!(strand, Strand::Forward);
    sifter.scan(&positions, wt_freq, mut_freq, forward);

    let mut candidate_sites = sifter.candidate_sites;
    candidate_sites.sort();
    candidate_sites.dedup();
    Ok(candidate_sites)
}

/// Find conversion sites over a membership-restricted cell set.
///
/// Used by both m6A (WT-vs-MUT) and A-to-I (control-free, reference-anchored)
/// whenever `--cell-membership` supplies one. Reads WT BAM files once over the
/// membership cells, then discovers on their pooled marginal — NOT per cell
/// type: a putative site needs only the motif and observed WT C→U at/above the
/// coverage floors, so the pooled marginal detects every site any one type
/// would, with the full WT evidence per site. (The cell types drive the
/// second-pass quantification; the effect-size call is the downstream per-site
/// test, so per-type discovery would only under-count each site's 2×2.)
fn find_sites_with_celltype_stats(
    gff_record: &GffRecord,
    params: &ConversionParams,
    faidx_reader: &faidx::Reader,
    cache: &mut crate::data::bam_io::BamReaderCache,
    chr: &str,
    strand: &Strand,
    membership: &CellMembership,
) -> anyhow::Result<Vec<ConversionSite>> {
    let cell_types = membership.cell_types();

    if cell_types.is_empty() {
        return Ok(Vec::new());
    }

    // Read WT BAM files ONCE, tracking per-cell frequencies
    let mut wt_per_cell_map =
        DnaBaseFreqMap::new_with_cell_barcode(&params.cell_barcode_tag, Some(membership));
    wt_per_cell_map.set_quality_thresholds(params.min_base_quality, params.min_mapping_quality);
    params.apply_umi(&mut wt_per_cell_map);
    for wt_file in &params.wt_bam_files {
        params.restrict_to_competent_cells(&mut wt_per_cell_map, wt_file);
        wt_per_cell_map.update_from_gene_cached(
            cache,
            wt_file,
            gff_record,
            &params.gene_barcode_tag,
            params.include_missing_barcode,
        )?;
    }

    if wt_per_cell_map.num_positions() < params.min_length_for_testing() {
        return Ok(Vec::new());
    }

    let forward = matches!(strand, Strand::Forward);

    // Pooled MUT (control) background for the m6A contrast — the control is not
    // stratified by cell type, so it is built once here and shared across all
    // cell types (mirrors `find_sites_with_bulk_stats`). Empty ⇒ `None`: A-to-I
    // has no control arm, so it falls through to the reference-anchored
    // beta-binomial null per stratum.
    let mut mut_base_freq_map = DnaBaseFreqMap::new();
    let mut_freq = if params.mut_bam_files.is_empty() {
        None
    } else {
        mut_base_freq_map
            .set_quality_thresholds(params.min_base_quality, params.min_mapping_quality);
        params.apply_umi(&mut mut_base_freq_map);
        for mut_file in &params.mut_bam_files {
            mut_base_freq_map.update_from_gene_cached(
                cache,
                mut_file,
                gff_record,
                &params.gene_barcode_tag,
                params.include_missing_barcode,
            )?;
        }
        mut_base_freq_map.marginal_frequency_map()
    };

    // Discover on the ALL-cell WT marginal (pooled over this gene's membership
    // cells), NOT per cell type. A putative site needs only the motif and
    // observed WT C→U at/above the coverage floors, and total coverage /
    // conversion ≥ any single type's — so the pooled marginal detects every
    // site any one type would and each candidate carries the gene's full WT
    // evidence. Per-type scanning here would emit one site per (position, type)
    // and, after dedup, leave each site's 2×2 counting only one type's reads —
    // silently under-calling edited sites. It also turns discovery into a max
    // over types, which this pipeline cannot absorb: it runs one exact test per
    // site with no multiplicity correction. The effect-size decision is the
    // downstream per-site test; the cell types drive the second pass, not this.
    let total_wt = wt_per_cell_map.pooled_frequency_map();
    let mut positions: Vec<i64> = total_wt.keys().copied().collect();
    positions.sort_unstable();
    if positions.len() < params.min_length_for_testing() {
        return Ok(Vec::new());
    }

    let mut sifter = params.create_sifter(faidx_reader, chr, positions.len());
    sifter.scan(&positions, &total_wt, mut_freq, forward);
    let mut all_candidate_sites = sifter.candidate_sites;

    all_candidate_sites.sort();
    all_candidate_sites.dedup();
    Ok(all_candidate_sites)
}

//////////////////////////////////////////
// SECOND PASS: Collect cell-level data //
//////////////////////////////////////////

/// Gather conversion statistics for all sites in all genes from a single BAM file.
pub fn gather_conversion_stats(
    gene_sites: &HashMap<GeneId, Vec<ConversionSite>>,
    params: &ConversionParams,
    gff_map: &GffRecordMap,
    bam_file: &str,
    cell_membership: Option<&CellMembership>,
    valid_cell_barcodes: Option<&rustc_hash::FxHashSet<CellBarcode>>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, ConversionData)>> {
    let ndata = gene_sites.iter().map(|x| x.value().len()).sum::<usize>();
    let arc_ret = Arc::new(Mutex::new(Vec::with_capacity(ndata)));

    gene_sites
        .into_iter()
        .par_bridge()
        .progress_with(new_progress_bar(gene_sites.len() as u64))
        .try_for_each_init(
            crate::data::bam_io::BamReaderCache::new,
            |cache, gs| -> anyhow::Result<()> {
                let gene = gs.key();
                let sites = gs.value();

                if let Some(gff) = gff_map.get(gene) {
                    let stats = collect_gene_conversion_stats(
                        cache,
                        params,
                        bam_file,
                        &gff,
                        sites,
                        cell_membership,
                    )?;
                    arc_ret.lock().expect("lock").extend(stats);
                }
                Ok(())
            },
        )?;

    let mut stats = Arc::try_unwrap(arc_ret)
        .map_err(|_| anyhow::anyhow!("failed to release stats"))?
        .into_inner()?;

    if let Some(valid_cells) = valid_cell_barcodes {
        let before = stats.len();
        stats.retain(|(cb, _, _)| valid_cells.contains(cb));
        info!(
            "filtered to QC-passing cells: {} -> {} conversion stats",
            before,
            stats.len()
        );
    }

    Ok(stats)
}

/// Extract conversion statistics for a single site from a pre-loaded DnaBaseFreqMap.
///
/// This is used by the optimized `collect_gene_conversion_stats()` which reads the
/// gene region once and queries each site from the cached map.
fn extract_site_stats_from_map(
    params: &ConversionParams,
    gff_record: &GffRecord,
    site: &ConversionSite,
    stat_map: &DnaBaseFreqMap,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, ConversionData)>> {
    let gene = gff_record.gene_id.clone();
    let chr = gff_record.seqname.as_ref();
    let strand = gff_record.strand;

    match site {
        ConversionSite::M6A {
            m6a_pos,
            conversion_pos,
            ..
        } => {
            let conversion_stat = stat_map.stratified_frequency_at(*conversion_pos);

            let Some(conv_stat) = conversion_stat else {
                return Ok(Vec::new());
            };

            // M6A: C->T (forward strand) or G->A (reverse strand)
            let (unmutated_base, mutated_base) = match strand {
                Strand::Forward => (Dna::C, Dna::T),
                Strand::Backward => (Dna::G, Dna::A),
            };

            let (start, stop) = (*conversion_pos, *conversion_pos + 1);

            let stats = conv_stat
                .iter()
                .filter_map(|(cb, counts)| {
                    let converted = counts.get(Some(&mutated_base));
                    let unconverted = counts.get(Some(&unmutated_base));

                    if (params.include_missing_barcode || cb != &CellBarcode::Missing)
                        && converted > 0
                    {
                        Some((
                            cb.clone(),
                            BedWithGene {
                                chr: chr.into(),
                                start,
                                stop,
                                gene: gene.clone(),
                                strand,
                            },
                            ConversionData {
                                converted,
                                unconverted,
                                site_pos: *m6a_pos,
                            },
                        ))
                    } else {
                        None
                    }
                })
                .collect();

            Ok(stats)
        }
        ConversionSite::AtoI { editing_pos, .. } => {
            let conversion_stat = stat_map.stratified_frequency_at(*editing_pos);

            let Some(conv_stat) = conversion_stat else {
                return Ok(Vec::new());
            };

            // A-to-I: A->G (forward strand) or T->C (reverse strand)
            let (unmutated_base, mutated_base) = match strand {
                Strand::Forward => (Dna::A, Dna::G),
                Strand::Backward => (Dna::T, Dna::C),
            };

            let (start, stop) = (*editing_pos, *editing_pos + 1);

            let stats = conv_stat
                .iter()
                .filter_map(|(cb, counts)| {
                    let edited = counts.get(Some(&mutated_base));
                    let unedited = counts.get(Some(&unmutated_base));

                    if (params.include_missing_barcode || cb != &CellBarcode::Missing) && edited > 0
                    {
                        Some((
                            cb.clone(),
                            BedWithGene {
                                chr: chr.into(),
                                start,
                                stop,
                                gene: gene.clone(),
                                strand,
                            },
                            ConversionData {
                                converted: edited,
                                unconverted: unedited,
                                site_pos: *editing_pos,
                            },
                        ))
                    } else {
                        None
                    }
                })
                .collect();

            Ok(stats)
        }
    }
}

fn collect_gene_conversion_stats(
    cache: &mut crate::data::bam_io::BamReaderCache,
    params: &ConversionParams,
    bam_file: &str,
    gff_record: &GffRecord,
    sites: &[ConversionSite],
    cell_membership: Option<&CellMembership>,
) -> anyhow::Result<Vec<(CellBarcode, BedWithGene, ConversionData)>> {
    if sites.is_empty() {
        return Ok(Vec::new());
    }

    // OPTIMIZATION: Read gene region ONCE for all sites instead of once per site.
    // This eliminates massive I/O overhead (30-50x speedup for genes with many sites).
    // Additionally, only store frequencies for the specific site positions to minimize memory.
    let mut stat_map =
        DnaBaseFreqMap::new_with_cell_barcode(&params.cell_barcode_tag, cell_membership);
    stat_map.set_quality_thresholds(params.min_base_quality, params.min_mapping_quality);
    params.apply_umi(&mut stat_map);

    // Collect all positions we need to query and calculate minimal region bounds
    let mut positions_to_track = rustc_hash::FxHashSet::default();
    let (min_pos, max_pos) = sites.iter().fold((i64::MAX, i64::MIN), |(min, max), site| {
        match site {
            ConversionSite::M6A {
                m6a_pos,
                conversion_pos,
                ..
            } => {
                // For M6A, we need both the m6A position and conversion position
                positions_to_track.insert(*m6a_pos);
                positions_to_track.insert(*conversion_pos);
                let site_min = (*m6a_pos).min(*conversion_pos);
                let site_max = (*m6a_pos).max(*conversion_pos);
                (min.min(site_min), max.max(site_max))
            }
            ConversionSite::AtoI { editing_pos, .. } => {
                // For ATOI, we only need the editing position
                positions_to_track.insert(*editing_pos);
                (min.min(*editing_pos), max.max(*editing_pos))
            }
        }
    });

    // Set position filter to only store frequencies for these specific positions
    stat_map.set_position_filter(positions_to_track);

    // Create a minimal GFF record that covers only the sites region (not entire gene)
    let mut minimal_gff = gff_record.clone();
    minimal_gff.start = (min_pos - BAM_READ_PADDING).max(0);
    minimal_gff.stop = max_pos + BAM_READ_PADDING;

    // Read only the minimal region spanning all sites, but only accumulate for tracked positions
    stat_map.update_from_gene_cached(
        cache,
        bam_file,
        &minimal_gff,
        &params.gene_barcode_tag,
        params.include_missing_barcode,
    )?;

    // Extract stats for each site from pre-loaded map
    let mut all_stats = Vec::new();

    for site in sites {
        let stats = extract_site_stats_from_map(params, gff_record, site, &stat_map)?;
        all_stats.extend(stats);
    }

    Ok(all_stats)
}

////////////////////
// Backend output //
////////////////////

/// Row-union + backend-file accumulator for one output resolution, collected
/// across all batches so rows can be reordered to a shared union before each
/// staging directory is finalized. Editing emits two resolutions in parallel:
/// the gene-level pooled matrix and the per-site matrix.
#[derive(Default)]
struct ResolutionOutputs {
    rows: HashSet<Box<str>>,
    files: Vec<crate::quant::BackendOutputPath>,
}

/// Unified backend output for conversion sites (m6A or A-to-I).
///
/// Emits two matrices per batch from the SAME single BAM pass:
/// - `{batch}_{modality}`       — gene-level pooled `(positive, coverage)` rows,
///   the co-embedding form (see [`summarize_stats_two_channel`]).
/// - `{batch}_{modality}_site`  — per-site rows keyed on the single-base
///   `{chr}:{pos}` subunit (see [`summarize_stats_per_site`]).
pub fn process_all_bam_files_to_backend(
    params: &ConversionParams,
    gene_sites: &HashMap<GeneId, Vec<ConversionSite>>,
    gff_map: &GffRecordMap,
    valid_cell_barcodes: Option<
        &rustc_hash::FxHashMap<Box<str>, rustc_hash::FxHashSet<CellBarcode>>,
    >,
) -> anyhow::Result<()> {
    let membership = params.load_membership()?;

    let gene_key = create_gene_key_function(gff_map);

    let ctx = BamProcessContext {
        gene_sites,
        params,
        gff_map,
        gene_key: &gene_key,
        cell_membership: membership.as_ref(),
        valid_cell_barcodes,
    };

    let mut gene_out = ResolutionOutputs::default();
    let mut site_out = ResolutionOutputs::default();

    // Quantify the discovered sites in EVERY sample, both WT and MUT (control)
    // See `quant_bam_files`.
    let all_bam_files = params.quant_bam_files();
    let batch_names = uniq_batch_names(&all_bam_files)?;

    for (bam_file, batch_name) in all_bam_files.iter().zip(batch_names) {
        process_bam_to_backend(bam_file, &batch_name, &ctx, &mut gene_out, &mut site_out)?;
    }

    // Log match statistics if membership was used
    if let Some(ref m) = membership {
        let (matched, total) = m.match_stats();
        info!(
            "Cell barcode matching: {}/{} BAM barcodes matched membership ({:.1}%)",
            matched,
            total,
            if total > 0 {
                100.0 * matched as f32 / total as f32
            } else {
                0.0
            }
        );
    }

    // Reorder rows to ensure consistency across files (per resolution — the
    // gene-level and per-site matrices have disjoint row vocabularies).
    reorder_all_matrices(params, gene_out)?;
    reorder_all_matrices(params, site_out)?;

    Ok(())
}

/// Shared immutable context for BAM-to-backend processing.
struct BamProcessContext<'a, GK: Fn(&BedWithGene) -> Box<str> + Send + Sync> {
    gene_sites: &'a HashMap<GeneId, Vec<ConversionSite>>,
    params: &'a ConversionParams,
    gff_map: &'a GffRecordMap,
    gene_key: &'a GK,
    cell_membership: Option<&'a CellMembership>,
    valid_cell_barcodes:
        Option<&'a rustc_hash::FxHashMap<Box<str>, rustc_hash::FxHashSet<CellBarcode>>>,
}

fn process_bam_to_backend(
    bam_file: &str,
    batch_name: &str,
    ctx: &BamProcessContext<'_, impl Fn(&BedWithGene) -> Box<str> + Send + Sync>,
    gene_out: &mut ResolutionOutputs,
    site_out: &mut ResolutionOutputs,
) -> anyhow::Result<()> {
    info!(
        "collecting data over {} sites from {} ...",
        ctx.gene_sites
            .iter()
            .map(|x| x.value().len())
            .sum::<usize>(),
        bam_file
    );

    // Each batch is filtered by ITS OWN called cell set (per-library knee),
    // looked up by BAM file path (stable across the QC and quant passes).
    let batch_valid_cells = ctx.valid_cell_barcodes.and_then(|m| m.get(bam_file));

    let stats = gather_conversion_stats(
        ctx.gene_sites,
        ctx.params,
        ctx.gff_map,
        bam_file,
        ctx.cell_membership,
        batch_valid_cells,
    )?;

    let (modality, pos_channel, neg_channel) = match ctx.params.mod_type {
        ModificationType::M6A { .. } => (
            data_beans::aux::feature_rows::M6A,
            data_beans::aux::feature_rows::METHYLATED,
            data_beans::aux::feature_rows::UNMETHYLATED,
        ),
        ModificationType::AtoI => (
            data_beans::aux::feature_rows::ATOI,
            data_beans::aux::feature_rows::EDITED,
            data_beans::aux::feature_rows::UNEDITED,
        ),
    };

    info!(
        "aggregating {} site stats → gene-level + per-site two-channel (edited + coverage)...",
        stats.len()
    );

    // Gene-level: pool sites → gene, both channels as rows in `{batch}_{modality}`:
    // `{gene}/{modality}/{pos}` = Σ converted, `{gene}/{modality}/{neg}` = Σ
    // unconverted — the gene-per-channel `(positive, coverage)` co-embedding form.
    let gene_triplets =
        summarize_stats_two_channel(&stats, ctx.gene_key, modality, pos_channel, neg_channel);
    write_resolution_backend(ctx, batch_name, modality, gene_triplets, gene_out)?;

    // Per-site: same channels keyed on the single-base `{chr}:{pos}` subunit, in
    // `{batch}_{modality}_site`: `{gene}/{modality}/{chr}:{pos}/{pos,neg}`.
    let site_triplets =
        summarize_stats_per_site(&stats, ctx.gene_key, modality, pos_channel, neg_channel);
    write_resolution_backend(
        ctx,
        batch_name,
        &format!("{}_site", modality),
        site_triplets,
        site_out,
    )?;

    Ok(())
}

/// Write one resolution's triplets to a `{batch}_{name_suffix}` backend and
/// record its rows + deferred finalize into `out`. Finalize is deferred because
/// rows are later reordered to the shared cross-batch union (see
/// [`reorder_all_matrices`]). No post-write nnz squeeze: the gene axis is QC'd
/// upstream (`gene_min_cells`), every putative site is kept by design, and
/// cells are frozen by the shared cell call — `faba qc` is where a squeeze
/// belongs.
fn write_resolution_backend(
    ctx: &BamProcessContext<'_, impl Fn(&BedWithGene) -> Box<str> + Send + Sync>,
    batch_name: &str,
    name_suffix: &str,
    triplets: TripletsRowsCols,
    out_acc: &mut ResolutionOutputs,
) -> anyhow::Result<()> {
    let out = ctx
        .params
        .backend_output_path(&format!("{}_{}", batch_name, name_suffix));
    let data = triplets.to_backend(&out.write_path)?;
    out_acc.rows.extend(data.row_names()?);
    info!("created {name_suffix} data: {}", &out.target_path);
    drop(data);
    out_acc.files.push(out);
    Ok(())
}

/// Reorder rows in-place across all per-batch backends of one resolution so row
/// indices are consistent across files, then finalize each staging directory
/// into its `.zarr.zip` archive (no-op for `.h5` or unzipped `.zarr`).
fn reorder_all_matrices(params: &ConversionParams, out: ResolutionOutputs) -> anyhow::Result<()> {
    let backend = &params.backend;

    let mut rows_sorted: Vec<_> = out.rows.into_iter().collect();
    rows_sorted.sort();

    for out in &out.files {
        open_sparse_matrix(&out.write_path, backend)?.reorder_rows(&rows_sorted)?;
    }

    for out in out.files {
        out.finalize()?;
    }

    Ok(())
}

impl ModificationType {
    /// Human-readable label for log messages
    fn label(&self) -> &'static str {
        match self {
            ModificationType::M6A { .. } => "m6A",
            ModificationType::AtoI => "A-to-I",
        }
    }
}

#[cfg(test)]
mod tests;
