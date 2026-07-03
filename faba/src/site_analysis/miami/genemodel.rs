//! Gene-model middle track for the Miami figure.
//!
//! Loads the selected gene(s) from a GTF/GFF — gene footprint + strand
//! (via `build_gene_map`) and merged exons (via `build_exon_intervals`) —
//! and emits an SVG band of exon rectangles on an intron line with
//! strand-direction chevrons. The x-axis is purely positional (never
//! reversed on the minus strand); only the chevrons flip direction, so
//! the model stays aligned with the matrix/depth tracks above and below.

use super::bin::BinEdges;
use crate::site_analysis::pileup::Selector;
use genomic_data::gff::{
    build_exon_intervals, build_gene_map, read_gff_record_vec, FeatureType, GeneSymbol,
};
use genomic_data::sam::Strand;
use plot_utils::svg_emit::escape_xml;
use std::fmt::Write as _;

/// One gene resolved from the GTF.
pub struct GeneModel {
    pub chr: Box<str>,
    /// 0-based half-open gene footprint `[lo, hi)`.
    pub lo: i64,
    pub hi: i64,
    pub forward: bool,
    /// 0-based half-open merged exons, sorted.
    pub exons: Vec<(i64, i64)>,
    pub symbol: Box<str>,
}

/// Load the gene model(s) matching `selector` from the GTF. Matching
/// reuses the same `{gene_id}_{symbol}` key + relaxed canonicalizer the
/// matrix rows use, so a `-q BRCA2` query resolves the GTF gene too.
pub fn load_gene_models(gtf: &str, selector: &Selector) -> anyhow::Result<Vec<GeneModel>> {
    let records = read_gff_record_vec(gtf)?;
    let gene_map = build_gene_map(&records, Some(&FeatureType::Gene))?;
    let exon_map = build_exon_intervals(&records);

    let mut out = Vec::new();
    for entry in gene_map.iter() {
        let gene_id = entry.key();
        let rec = entry.value();

        let (symbol, gene_key): (Box<str>, Box<str>) = match &rec.gene_name {
            GeneSymbol::Symbol(s) if !s.is_empty() => {
                (s.clone(), format!("{}_{}", gene_id, s).into())
            }
            _ => {
                let id: Box<str> = gene_id.to_string().into();
                (id.clone(), id)
            }
        };

        if !selector.matches_gene(&gene_key) {
            continue;
        }

        let exons = exon_map
            .get(gene_id)
            .map(|e| e.value().clone())
            .unwrap_or_default();

        out.push(GeneModel {
            chr: rec.seqname.clone(),
            lo: rec.start - 1, // GTF 1-based inclusive -> 0-based half-open
            hi: rec.stop,
            forward: matches!(rec.strand, Strand::Forward),
            exons,
            symbol,
        });
    }
    // Stable order (smallest start first) for deterministic rendering.
    out.sort_by_key(|g| (g.lo, g.hi));
    Ok(out)
}

/// Genomic extent spanned by a set of gene models (0-based half-open),
/// or `None` if empty / no usable coordinates.
pub fn models_extent(models: &[GeneModel]) -> Option<(Box<str>, i64, i64)> {
    let mut it = models.iter();
    let first = it.next()?;
    let mut lo = first.lo;
    let mut hi = first.hi;
    for g in it {
        lo = lo.min(g.lo);
        hi = hi.max(g.hi);
    }
    Some((first.chr.clone(), lo, hi))
}

/// Emit the SVG for one gene model into `[x_left, x_left + plot_w]`,
/// centered on `y_mid`. `band_h` is the exon-rectangle height.
pub fn gene_model_svg(
    g: &GeneModel,
    edges: &BinEdges,
    x_left: f32,
    plot_w: f32,
    y_mid: f32,
    band_h: f32,
) -> String {
    let mut s = String::with_capacity(1024);
    let x0 = edges.x_px(g.lo, x_left, plot_w);
    let x1 = edges.x_px(g.hi, x_left, plot_w);

    // Intron line (gene footprint).
    let _ = write!(
        s,
        "<line x1=\"{x0:.1}\" y1=\"{y_mid:.1}\" x2=\"{x1:.1}\" y2=\"{y_mid:.1}\" \
         stroke=\"#666\" stroke-width=\"1.2\"/>"
    );

    // Strand-direction chevrons along the intron line.
    let step = (band_h * 1.4).max(8.0);
    let head = (band_h * 0.35).max(2.0);
    let mut x = x0 + step * 0.5;
    while x < x1 - 1.0 {
        let (tipx, basex) = if g.forward {
            (x + head, x - head)
        } else {
            (x - head, x + head)
        };
        let _ = write!(
            s,
            "<polyline points=\"{basex:.1},{:.1} {tipx:.1},{y_mid:.1} {basex:.1},{:.1}\" \
             fill=\"none\" stroke=\"#666\" stroke-width=\"1.0\"/>",
            y_mid - head,
            y_mid + head
        );
        x += step;
    }

    // Exon rectangles.
    let ytop = y_mid - band_h / 2.0;
    for &(es, ee) in &g.exons {
        if ee <= edges.min_pos || es > edges.max_pos {
            continue;
        }
        let ex0 = edges.x_px(es, x_left, plot_w);
        let ex1 = edges.x_px(ee, x_left, plot_w);
        let w = (ex1 - ex0).max(0.8);
        let _ = write!(
            s,
            "<rect x=\"{ex0:.1}\" y=\"{ytop:.1}\" width=\"{w:.1}\" height=\"{band_h:.1}\" \
             fill=\"#888\" stroke=\"#444\" stroke-width=\"0.4\"/>"
        );
    }

    // Symbol label, just left of the gene start (or clamped into view).
    let lx = (x0 - 2.0).max(x_left);
    let _ = write!(
        s,
        "<text x=\"{lx:.1}\" y=\"{:.1}\" text-anchor=\"end\" font-family=\"sans-serif\" \
         font-size=\"{:.1}\" fill=\"#333\">{}</text>",
        y_mid + band_h * 0.35,
        band_h * 0.9,
        escape_xml(&g.symbol)
    );

    s
}

#[cfg(test)]
mod tests;
