//! Gene placement + per-strand binning for `senna plot-strand`.
//!
//! Resolves each activity-matrix gene to a (chromosome, bin, strand) via
//! a GTF, then accumulates per-group activity into Watson(forward) and
//! Crick(reverse) bin sums — the signal the renderer mirrors.

use super::PlotStrandArgs;
use crate::embed_common::Mat;
use auxiliary_data::feature_names::FeatureNameKind;
use genomic_data::coordinates::{chr_stripped, load_gene_loci_map, GeneLoc};
use genomic_data::sam::Strand;
use rustc_hash::FxHashMap;

////////////////////////////////////////////////////////////////////////
// Gene placement (coordinate + strand + bin geometry)
////////////////////////////////////////////////////////////////////////

/// Per-gene placement onto the (chromosome, bin) grid.
pub(super) struct Placement {
    pub(super) gene_idx: usize,
    pub(super) chr_idx: usize,
    pub(super) bin: usize,
    pub(super) forward: bool,
    pub(super) tss: i64,
    /// HGNC symbol from the matched GTF `gene_name` (falls back to the
    /// GENCODE clone name / Ensembl id when no symbol is annotated).
    /// Used for labels so we show the canonical symbol, not the raw
    /// `ENSG…_SYMBOL` row name.
    pub(super) symbol: Box<str>,
}

/// One chromosome's binning geometry.
pub(super) struct ChrGeom {
    pub(super) name: Box<str>,
    pub(super) min_pos: i64,
    pub(super) span: i64,
    pub(super) n_bins: usize,
}

pub(super) struct Placements {
    pub(super) chromosomes: Vec<ChrGeom>,
    pub(super) placed: Vec<Placement>,
    /// Max chromosome span (bp) — drives proportional widths.
    pub(super) max_span: i64,
}

/// A gene matched to the GTF: activity-matrix index + resolved location,
/// before chromosome geometry (and thus bins) is known.
struct Hit {
    gene_idx: usize,
    chr: Box<str>,
    tss: i64,
    forward: bool,
    symbol: Box<str>,
}

pub(super) fn place_genes(
    gene_names: &[Box<str>],
    gtf: &str,
    args: &PlotStrandArgs,
) -> anyhow::Result<Placements> {
    // GTF symbol -> (gtf_symbol, chr, tss, strand), canonicalized for
    // alias-tolerant matching against the activity row names (e.g.
    // ENSG…_TGFB1 → TGFB1). The original GTF `gene_name` is kept so labels
    // can show the canonical HGNC symbol rather than the raw row name.
    let raw_map = load_gene_loci_map(gtf)?;
    let kind = FeatureNameKind::Gene { delim: '_' };
    let mut canon_map: FxHashMap<Box<str>, (Box<str>, GeneLoc)> = FxHashMap::default();
    for (sym, loc) in raw_map {
        canon_map
            .entry(kind.canonicalize(&sym))
            .or_insert((sym, loc));
    }

    // Optional chromosome whitelist (strip "chr").
    let allow: Option<Vec<Box<str>>> = args.chromosomes.as_ref().map(|v| {
        v.iter()
            .map(|c| chr_stripped(c).to_string().into_boxed_str())
            .collect()
    });
    let allowed = |chr: &str| -> bool {
        allow
            .as_ref()
            .is_none_or(|a| a.iter().any(|c| c.as_ref() == chr))
    };

    // A matched gene: activity-matrix index + its resolved location.
    let mut hits: Vec<Hit> = Vec::new();
    for (gi, name) in gene_names.iter().enumerate() {
        let Some((sym, loc)) = canon_map.get(&kind.canonicalize(name)) else {
            continue;
        };
        let chr = chr_stripped(&loc.chr);
        if !allowed(chr) {
            continue;
        }
        hits.push(Hit {
            gene_idx: gi,
            chr: chr.to_string().into_boxed_str(),
            tss: loc.tss,
            forward: matches!(loc.strand, Strand::Forward),
            symbol: sym.clone(),
        });
    }

    // Per-chromosome min/max positions → spans.
    let mut span_by_chr: FxHashMap<Box<str>, (i64, i64)> = FxHashMap::default();
    for h in &hits {
        let e = span_by_chr.entry(h.chr.clone()).or_insert((h.tss, h.tss));
        e.0 = e.0.min(h.tss);
        e.1 = e.1.max(h.tss);
    }

    // Karyotype ordering: 1..22, X, Y, M, then anything else lexically.
    let mut chr_list: Vec<Box<str>> = span_by_chr.keys().cloned().collect();
    chr_list.sort_by_key(|c| chr_sort_key(c));

    let max_span = span_by_chr
        .values()
        .map(|(lo, hi)| (hi - lo).max(1))
        .max()
        .unwrap_or(1);
    let bin_bp = (max_span as f64 / args.bins.max(1) as f64).max(1.0);

    let mut chr_idx: FxHashMap<Box<str>, usize> = FxHashMap::default();
    let mut chromosomes: Vec<ChrGeom> = Vec::with_capacity(chr_list.len());
    for (ci, chr) in chr_list.iter().enumerate() {
        let (lo, hi) = span_by_chr[chr];
        let span = (hi - lo).max(1);
        let n_bins = ((span as f64 / bin_bp).ceil() as usize).max(1);
        chr_idx.insert(chr.clone(), ci);
        chromosomes.push(ChrGeom {
            name: chr.clone(),
            min_pos: lo,
            span,
            n_bins,
        });
    }

    let placed: Vec<Placement> = hits
        .into_iter()
        .map(|h| {
            let ci = chr_idx[&h.chr];
            let g = &chromosomes[ci];
            let bin = (((h.tss - g.min_pos) as f64 / bin_bp).floor() as usize).min(g.n_bins - 1);
            Placement {
                gene_idx: h.gene_idx,
                chr_idx: ci,
                bin,
                forward: h.forward,
                tss: h.tss,
                symbol: h.symbol,
            }
        })
        .collect();

    Ok(Placements {
        chromosomes,
        placed,
        max_span,
    })
}

/// Sort key: autosomes by number, then X, Y, M/MT, then other.
fn chr_sort_key(chr: &str) -> (u8, u32, Box<str>) {
    if let Ok(n) = chr.parse::<u32>() {
        (0, n, chr.into())
    } else {
        match chr.to_ascii_uppercase().as_str() {
            "X" => (1, 0, chr.into()),
            "Y" => (1, 1, chr.into()),
            "M" | "MT" => (1, 2, chr.into()),
            _ => (2, 0, chr.into()),
        }
    }
}

////////////////////////////////////////////////////////////////////////
// Binning
////////////////////////////////////////////////////////////////////////

/// Per-chromosome Watson(up)/Crick(down) bin sums for one group.
pub(super) struct BinGrid {
    pub(super) watson: Vec<Vec<f32>>,
    pub(super) crick: Vec<Vec<f32>>,
}

impl BinGrid {
    fn empty(p: &Placements) -> BinGrid {
        BinGrid {
            watson: p.chromosomes.iter().map(|c| vec![0.0; c.n_bins]).collect(),
            crick: p.chromosomes.iter().map(|c| vec![0.0; c.n_bins]).collect(),
        }
    }

    pub(super) fn iter_values(&self) -> impl Iterator<Item = f32> + '_ {
        self.watson
            .iter()
            .chain(self.crick.iter())
            .flat_map(|v| v.iter().copied())
    }

    /// The bin vector for `pl`'s chromosome on its own strand.
    fn strand(&self, pl: &Placement) -> &[f32] {
        let s = if pl.forward {
            &self.watson
        } else {
            &self.crick
        };
        &s[pl.chr_idx]
    }

    fn strand_mut(&mut self, pl: &Placement) -> &mut Vec<f32> {
        let s = if pl.forward {
            &mut self.watson
        } else {
            &mut self.crick
        };
        &mut s[pl.chr_idx]
    }

    /// Accumulate `other` into `self` bin-for-bin (same geometry).
    fn add_assign(&mut self, other: &BinGrid) {
        for (a, b) in self.watson.iter_mut().zip(&other.watson) {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        }
        for (a, b) in self.crick.iter_mut().zip(&other.crick) {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        }
    }
}

pub(super) fn bin_group(mat: &Mat, col: usize, p: &Placements) -> BinGrid {
    let mut grid = BinGrid::empty(p);
    for pl in &p.placed {
        let v = mat[(pl.gene_idx, col)].max(0.0);
        if v > 0.0 {
            grid.strand_mut(pl)[pl.bin] += v;
        }
    }
    grid
}

/// Consensus = activity summed across all cell types. Equal to the
/// bin-wise sum of the per-group grids (binning is linear), so we fold
/// the already-computed grids rather than re-scanning the matrix.
pub(super) fn sum_grids(grids: &[BinGrid], p: &Placements) -> BinGrid {
    let mut acc = BinGrid::empty(p);
    for g in grids {
        acc.add_assign(g);
    }
    acc
}

/// 99th-percentile of positive values — a spike-robust scale so one
/// outlier bin doesn't flatten every panel. Quickselect (O(n)) rather
/// than a full sort.
pub(super) fn robust_max(values: impl Iterator<Item = f32>) -> f32 {
    let mut v: Vec<f32> = values.filter(|x| *x > 0.0).collect();
    if v.is_empty() {
        return 0.0;
    }
    let idx = (((v.len() - 1) as f32) * 0.99).round() as usize;
    *v.select_nth_unstable_by(idx, |a, b| a.partial_cmp(b).unwrap())
        .1
}

/// Height of the bin a placement falls into, on its own strand.
pub(super) fn bin_height(grid: &BinGrid, pl: &Placement) -> f32 {
    grid.strand(pl).get(pl.bin).copied().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("strandplace_{}_{}", std::process::id(), name))
    }

    /// Minimal args for placement tests (bins=10, defaults otherwise).
    fn args_with(gtf: &str) -> PlotStrandArgs {
        PlotStrandArgs {
            from: None,
            gtf: gtf.into(),
            activity: None,
            out: None,
            chromosomes: None,
            bins: 10,
            top_genes: 0,
            scale: crate::postprocess::plot::strand::HeightScale::Sqrt,
            consensus: true,
            watson_color: "#E69F00".into(),
            crick_color: "#0F8B8D".into(),
            consensus_color: "#7A7A7A".into(),
            width: 4.0,
            track_height: 0.3,
            dpi: 96,
            svg: false,
            png: false,
            no_pdf: true,
        }
    }

    /// chr1: G0 (+,100), G1 (-,300), G2 (+,500); chr2: G3 (+,100), G4 (-,400).
    fn write_gtf() -> std::path::PathBuf {
        let gtf = "\
chr1\tT\tgene\t100\t150\t.\t+\t.\tgene_id \"E0\"; gene_name \"G0\"
chr1\tT\tgene\t300\t350\t.\t-\t.\tgene_id \"E1\"; gene_name \"G1\"
chr1\tT\tgene\t500\t550\t.\t+\t.\tgene_id \"E2\"; gene_name \"G2\"
chr2\tT\tgene\t100\t150\t.\t+\t.\tgene_id \"E3\"; gene_name \"G3\"
chr2\tT\tgene\t400\t450\t.\t-\t.\tgene_id \"E4\"; gene_name \"G4\"
";
        let p = tmp("genes.gtf");
        std::fs::write(&p, gtf).unwrap();
        p
    }

    #[test]
    fn placement_and_binning_split_by_strand() {
        let gtf = write_gtf();
        // Activity gene names use the ENSG_SYMBOL alias form to exercise
        // the canonicalizer (G0 → "ENSGX_G0").
        let genes: Vec<Box<str>> = (0..5).map(|i| format!("ENSGX_G{i}").into()).collect();
        // Single group; only forward genes G0,G2 (chr1) and G3 (chr2) get
        // weight, reverse G1,G4 get zero.
        let mut mat = Mat::zeros(5, 1);
        mat[(0, 0)] = 5.0; // G0 + chr1
        mat[(2, 0)] = 3.0; // G2 + chr1
        mat[(3, 0)] = 7.0; // G3 + chr2

        let args = args_with(gtf.to_str().unwrap());
        let p = place_genes(&genes, gtf.to_str().unwrap(), &args).unwrap();
        std::fs::remove_file(&gtf).ok();

        assert_eq!(p.chromosomes.len(), 2, "two chromosomes");
        assert_eq!(p.placed.len(), 5, "all five genes placed");

        let grid = bin_group(&mat, 0, &p);
        let chr1 = p.chromosomes.iter().position(|c| &*c.name == "1").unwrap();
        let chr2 = p.chromosomes.iter().position(|c| &*c.name == "2").unwrap();

        let w1: f32 = grid.watson[chr1].iter().sum();
        let c1: f32 = grid.crick[chr1].iter().sum();
        let w2: f32 = grid.watson[chr2].iter().sum();
        let c2: f32 = grid.crick[chr2].iter().sum();

        assert!((w1 - 8.0).abs() < 1e-5, "chr1 Watson = G0+G2 = 8");
        assert_eq!(c1, 0.0, "chr1 Crick = 0 (G1 had no activity)");
        assert!((w2 - 7.0).abs() < 1e-5, "chr2 Watson = G3 = 7");
        assert_eq!(c2, 0.0, "chr2 Crick = 0 (G4 had no activity)");
    }
}
