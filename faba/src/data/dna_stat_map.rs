use crate::data::bam_io;
use crate::data::cell_membership::CellMembership;
use crate::data::dna::*;
use genomic_data::bed::Bed;
use genomic_data::gff::GffRecord;
use genomic_data::sam::*;

use rust_htslib::bam::{self, ext::BamRecordExtensions, record::Aux};

pub use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet;

// UMI hashing lives in `bam_io::hash_umi_bytes` — one definition, so the cell
// scan and the site counts dedup identically.

pub enum CountMode<'a> {
    Marginal {
        counts: HashMap<i64, DnaBaseCount>,
        // Per-read UMI dedup keyed by (cell, UMI hash), exactly as `PerCell`.
        // A UMI is only unique WITHIN a cell, so keying on the hash alone
        // collapses molecules from different cells and silently deletes
        // coverage. A read whose (cell, UMI) has been seen is skipped
        // entirely, not per-base.
        umi_seen: FxHashSet<(CellBarcode, u64)>,
    },
    PerCell {
        counts: HashMap<i64, HashMap<CellBarcode, DnaBaseCount>>,
        cell_barcode_tag: Vec<u8>,
        cell_membership: Option<&'a CellMembership>,
        // Per-read UMI dedup keyed by (cell, UMI hash).
        umi_seen: FxHashSet<(CellBarcode, u64)>,
    },
}

pub struct DnaBaseFreqMap<'a> {
    mode: CountMode<'a>,
    anchor: Option<(i64, Dna)>,
    position_filter: Option<FxHashSet<i64>>,
    min_base_quality: u8,
    min_mapping_quality: u8,
    umi_tag: Option<Vec<u8>>,
    /// Cell barcode tag used to key UMI dedup. Set together with `umi_tag` by
    /// [`Self::set_umi_tag`], because dedup is per (cell, UMI) in BOTH modes.
    dedup_cell_tag: Vec<u8>,
    use_base_quality: bool,
    quality_map: HashMap<i64, DnaBaseQual>,
    /// Restrict counting to these cells, in **either** [`CountMode`].
    ///
    /// `CountMode::PerCell`'s `cell_membership` cannot serve this purpose: it is
    /// absent from `Marginal`, so bulk discovery would silently ignore it. Site
    /// discovery on a de-diluted WT arm needs the filter to bite in `Marginal`
    /// mode without paying for a per-cell map, which is what this provides.
    keep_cells: Option<(Vec<u8>, std::sync::Arc<FxHashSet<CellBarcode>>)>,
}

impl<'a> Default for DnaBaseFreqMap<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> DnaBaseFreqMap<'a> {
    pub fn new() -> Self {
        Self {
            mode: CountMode::Marginal {
                counts: HashMap::default(),
                umi_seen: FxHashSet::default(),
            },
            anchor: None,
            position_filter: None,
            min_base_quality: 20,
            min_mapping_quality: 20,
            umi_tag: None,
            dedup_cell_tag: Vec::new(),
            use_base_quality: false,
            quality_map: HashMap::default(),
            keep_cells: None,
        }
    }

    /// Empty frequency map, keeping track of cell barcodes
    pub fn new_with_cell_barcode(
        cell_barcode_tag: &str,
        cell_membership: Option<&'a CellMembership>,
    ) -> Self {
        Self {
            mode: CountMode::PerCell {
                counts: HashMap::default(),
                cell_barcode_tag: cell_barcode_tag.as_bytes().to_vec(),
                cell_membership,
                umi_seen: FxHashSet::default(),
            },
            anchor: None,
            position_filter: None,
            min_base_quality: 20,
            min_mapping_quality: 20,
            umi_tag: None,
            dedup_cell_tag: Vec::new(),
            use_base_quality: false,
            quality_map: HashMap::default(),
            keep_cells: None,
        }
    }

    /// Set the anchor location and base for m6A counting:
    /// only count bases on reads that cover the anchor position with the expected base.
    #[allow(dead_code)]
    pub fn set_anchor_position(&mut self, loc: i64, base: Dna) {
        self.anchor = Some((loc, base));
    }

    /// Set position filter to only accumulate frequencies for specific positions.
    /// This dramatically reduces memory usage when processing many sites in a large gene.
    pub fn set_position_filter(&mut self, positions: FxHashSet<i64>) {
        self.position_filter = Some(positions);
    }

    /// Count only reads from `cells`, or from every cell when `None`. Applies in
    /// both count modes, so bulk (`Marginal`) discovery can be restricted to an
    /// already-called cell set — e.g. the editing-competent cells from
    /// [`crate::editing::cell_activity`]. Costs one barcode lookup per read,
    /// paid only when a set is installed.
    ///
    /// Takes an `Option` rather than pairing with a separate clear method
    /// because one map is REUSED across BAMs in the discovery loop: a library
    /// with no competent set must overwrite the previous library's, not inherit
    /// it. Making "no restriction" an argument means the caller cannot forget.
    pub fn set_keep_cells(
        &mut self,
        cell_barcode_tag: &str,
        cells: Option<std::sync::Arc<FxHashSet<CellBarcode>>>,
    ) {
        self.keep_cells = cells.map(|c| (cell_barcode_tag.as_bytes().to_vec(), c));
    }

    pub fn set_quality_thresholds(&mut self, min_base_quality: u8, min_mapping_quality: u8) {
        self.min_base_quality = min_base_quality;
        self.min_mapping_quality = min_mapping_quality;
    }

    /// Enable UMI-based deduplication, keyed by `(cell, UMI)` in BOTH modes.
    ///
    /// The cell barcode tag is required rather than optional because a 10x UMI
    /// is only unique within a droplet: two cells legitimately carry the same
    /// UMI sequence for different molecules, so collapsing on the hash alone
    /// deletes real coverage. `Marginal` did exactly that, which is why the
    /// tag is a parameter here instead of something a caller can forget.
    ///
    /// On data with no cell barcode the tag simply misses, every read resolves
    /// to `CellBarcode::Missing`, and the key degenerates to the UMI alone --
    /// the old behaviour, which is the right one for true bulk input.
    pub fn set_umi_tag(&mut self, umi_tag: &str, cell_barcode_tag: &str) {
        self.umi_tag = Some(umi_tag.as_bytes().to_vec());
        self.dedup_cell_tag = cell_barcode_tag.as_bytes().to_vec();
    }

    /// Enable per-base quality accumulation for Li 2011 genotype likelihood model.
    pub fn set_use_base_quality(&mut self, enabled: bool) {
        self.use_base_quality = enabled;
    }

    /// Reset UMI dedup state. Called at the start of every `update_from_*` so
    /// that dedup is scoped to a single BAM (matching 10x / umi_tools, which
    /// operate on one library at a time). Two BAMs that happen to share a
    /// `(cell, UMI)` value by coincidence are no longer collapsed into one.
    fn reset_umi_seen(&mut self) {
        match &mut self.mode {
            CountMode::Marginal { umi_seen, .. } => umi_seen.clear(),
            CountMode::PerCell { umi_seen, .. } => umi_seen.clear(),
        }
    }

    /// Update from BAM records in a BED region
    pub fn update_from_region(&mut self, bam_file_path: &str, region: &Bed) -> anyhow::Result<()> {
        self.reset_umi_seen();
        bam_io::for_each_record_in_region(bam_file_path, region, |_chr, rec| {
            self.add_bam_record(rec);
        })
    }

    /// Like [`Self::update_from_region`] but reuses a per-thread BAM reader cache.
    pub fn update_from_region_cached(
        &mut self,
        cache: &mut bam_io::BamReaderCache,
        bam_file_path: &str,
        region: &Bed,
    ) -> anyhow::Result<()> {
        self.reset_umi_seen();
        bam_io::for_each_record_in_region_cached(cache, bam_file_path, region, |_chr, rec| {
            self.add_bam_record(rec);
        })
    }

    /// Reuses a per-thread BAM reader cache; preferred inside `par_iter` paths.
    pub fn update_from_gene_cached(
        &mut self,
        cache: &mut bam_io::BamReaderCache,
        bam_file_path: &str,
        gff_record: &GffRecord,
        gene_barcode_tag: &str,
        include_missing_barcode: bool,
    ) -> anyhow::Result<()> {
        self.reset_umi_seen();
        bam_io::for_each_record_in_gene_cached(
            cache,
            bam_file_path,
            gff_record,
            gene_barcode_tag,
            include_missing_barcode,
            |rec| self.add_bam_record(rec),
        )
    }

    /// Extract UMI hash from a BAM record, if umi_tag is set.
    fn extract_umi_hash(&self, bam_record: &bam::Record) -> Option<u64> {
        let tag = self.umi_tag.as_ref()?;
        match bam_record.aux(tag) {
            Ok(Aux::String(s)) => Some(bam_io::hash_umi_bytes(s.as_bytes())),
            _ => None,
        }
    }

    fn add_bam_record(&mut self, bam_record: &bam::Record) {
        if !bam_io::passes_alignment_filters(bam_record, self.min_mapping_quality) {
            return;
        }
        // Interning a barcode is a thread-local borrow, a hash of the raw bytes
        // and an `Arc` clone — on EVERY alignment. The keep-set gate and the
        // per-cell counter read the same tag in every configuration faba builds,
        // so resolve it once and hand it on; if the tags ever differ, fall back
        // to re-extracting under the right one rather than silently keying the
        // counts off the wrong tag.
        let mut resolved: Option<CellBarcode> = None;
        if let Some((tag, keep)) = &self.keep_cells {
            let cb = bam_io::extract_cell_barcode(bam_record, tag);
            if !keep.contains(&cb) {
                return;
            }
            if matches!(&self.mode, CountMode::PerCell { cell_barcode_tag, .. }
                if cell_barcode_tag == tag)
            {
                resolved = Some(cb);
            }
        }

        let seq = bam_record.seq().as_bytes();
        let umi_hash = self.extract_umi_hash(bam_record);

        match &mut self.mode {
            CountMode::Marginal { counts, umi_seen } => {
                // Per-read dedup keyed by (cell, UMI): drop the whole record
                // before touching the per-base loop. Resolving the barcode here
                // costs a lookup only when dedup is on.
                if let Some(h) = umi_hash {
                    let cell = resolved.unwrap_or_else(|| {
                        bam_io::extract_cell_barcode(bam_record, &self.dedup_cell_tag)
                    });
                    if !umi_seen.insert((cell, h)) {
                        return;
                    }
                }
                let quals = bam_record.qual();

                for [rpos, gpos] in bam_record.aligned_pairs() {
                    if let Some(filter) = &self.position_filter {
                        if !filter.contains(&gpos) {
                            continue;
                        }
                    }

                    let phred = quals[rpos as usize];
                    if phred < self.min_base_quality {
                        continue;
                    }

                    let base = Dna::from_byte(seq[rpos as usize]);
                    counts.entry(gpos).or_default().add(base.as_ref(), 1);

                    if self.use_base_quality {
                        self.quality_map
                            .entry(gpos)
                            .or_default()
                            .add(base.as_ref(), phred);
                    }
                }
            }

            CountMode::PerCell {
                counts,
                cell_barcode_tag,
                cell_membership,
                umi_seen,
            } => {
                let cell_barcode = resolved
                    .unwrap_or_else(|| bam_io::extract_cell_barcode(bam_record, cell_barcode_tag));

                if let Some(membership) = cell_membership {
                    if membership.matches_barcode(&cell_barcode).is_none() {
                        return;
                    }
                }

                // Per-read dedup keyed by (cell, UMI): drop the whole record up front.
                if let Some(h) = umi_hash {
                    if !umi_seen.insert((cell_barcode.clone(), h)) {
                        return;
                    }
                }

                let anchor = self.anchor;
                let mut anchor_matched = anchor.is_none();
                let mut pending: Vec<(i64, Option<Dna>, u8)> = Vec::new();
                let quals = bam_record.qual();

                for [rpos, gpos] in bam_record.aligned_pairs() {
                    if let Some(filter) = &self.position_filter {
                        if !filter.contains(&gpos) {
                            continue;
                        }
                    }

                    let phred = quals[rpos as usize];
                    if phred < self.min_base_quality {
                        continue;
                    }

                    let base = Dna::from_byte(seq[rpos as usize]);
                    pending.push((gpos, base, phred));
                    if let Some((anchor_pos, anchor_base)) = anchor {
                        if gpos == anchor_pos && base.is_some_and(|b| b == anchor_base) {
                            anchor_matched = true;
                        }
                    }
                }

                if anchor_matched {
                    for (gpos, base, _phred) in pending {
                        let freq_map: &mut HashMap<CellBarcode, DnaBaseCount> =
                            counts.entry(gpos).or_default();
                        let freq = freq_map.entry(cell_barcode.clone()).or_default();
                        freq.add(base.as_ref(), 1);
                    }
                }
            }
        }
    }

    /// Output sorted genomic positions
    pub fn sorted_positions(&self) -> Vec<i64> {
        let mut ret: Vec<i64> = match &self.mode {
            CountMode::Marginal { counts, .. } => counts.keys().copied().collect(),
            CountMode::PerCell { counts, .. } => counts.keys().copied().collect(),
        };
        ret.sort();
        ret
    }

    /// Number of distinct genomic positions currently tracked — O(1), no
    /// allocation or sort (unlike [`Self::sorted_positions`]).
    pub fn num_positions(&self) -> usize {
        match &self.mode {
            CountMode::Marginal { counts, .. } => counts.len(),
            CountMode::PerCell { counts, .. } => counts.len(),
        }
    }

    /// Frequency stratified by cell barcodes on a genomic position `pos`
    pub fn stratified_frequency_at(&self, pos: i64) -> Option<&HashMap<CellBarcode, DnaBaseCount>> {
        match &self.mode {
            CountMode::PerCell { counts, .. } => counts.get(&pos),
            _ => None,
        }
    }

    /// Marginal frequency map
    pub fn marginal_frequency_map(&self) -> Option<&HashMap<i64, DnaBaseCount>> {
        match &self.mode {
            CountMode::Marginal { counts, .. } => Some(counts),
            _ => None,
        }
    }

    /// Pooled (all-cell) base counts per position, in **either** [`CountMode`].
    ///
    /// `Marginal` already stores exactly this, so it is borrowed; `PerCell`
    /// sums over cells. Site discovery needs the all-cell marginal even when
    /// the scan is stratified by cell — a per-group scan sees only that
    /// group's reads and would under-call — so it must not depend on the mode.
    /// Prefer this over [`Self::marginal_frequency_map`], which is `None` for
    /// `PerCell` and silently turns such a call into an error.
    pub fn pooled_frequency_map(&self) -> std::borrow::Cow<'_, HashMap<i64, DnaBaseCount>> {
        match &self.mode {
            CountMode::Marginal { counts, .. } => std::borrow::Cow::Borrowed(counts),
            CountMode::PerCell { counts, .. } => {
                let mut pooled: HashMap<i64, DnaBaseCount> = HashMap::default();
                pooled.reserve(counts.len());
                for (pos, by_cell) in counts.iter() {
                    let acc = pooled.entry(*pos).or_default();
                    for cell_count in by_cell.values() {
                        *acc += cell_count;
                    }
                }
                std::borrow::Cow::Owned(pooled)
            }
        }
    }

    /// Quality accumulator map (marginal mode only, requires set_use_base_quality(true))
    pub fn quality_map(&self) -> &HashMap<i64, DnaBaseQual> {
        &self.quality_map
    }
}

#[cfg(test)]
mod tests;
