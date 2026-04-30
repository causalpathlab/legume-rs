//! Shared output helpers: SVG/PDF/PNG emission, filename sanitization,
//! and the compact scientific-notation formatter used by the heatmap
//! colorbar.

use super::PlotTopicArgs;
use crate::embed_common::*;
use std::fs;
use std::path::Path;

pub(super) fn emit_outputs(
    svg: &str,
    w: u32,
    h: u32,
    base: &str,
    args: &PlotTopicArgs,
) -> anyhow::Result<()> {
    if args.svg {
        let path = format!("{base}.svg");
        fs::write(&path, svg.as_bytes())?;
        info!("Wrote {path}");
    }
    let png_task = args.png.then(|| format!("{base}.png"));
    let pdf_task = (!args.no_pdf).then(|| format!("{base}.pdf"));

    let (png_res, pdf_res) = rayon::join(
        || match &png_task {
            Some(p) => plot_utils::render_png(svg, w, h, Path::new(p)).map(|()| Some(p.clone())),
            None => Ok(None),
        },
        || match &pdf_task {
            Some(p) => plot_utils::render_pdf(svg, Path::new(p)).map(|()| Some(p.clone())),
            None => Ok(None),
        },
    );
    if let Some(p) = png_res? {
        info!("Wrote {p}");
    }
    if let Some(p) = pdf_res? {
        info!("Wrote {p}");
    }
    Ok(())
}

pub(super) fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' => c,
            _ => '_',
        })
        .collect()
}

/// Compact scientific-notation formatter for colorbar tick labels:
/// `2.7e-5` style with one fractional digit. Avoids the noise of
/// printf's `{:e}` (which uses `2.7e0` for 2.7) and is short enough to
/// fit a vertical colorbar's right margin.
pub(super) fn format_sci(v: f32) -> String {
    if !v.is_finite() || v == 0.0 {
        return "0".to_string();
    }
    let abs = v.abs();
    let exp = abs.log10().floor() as i32;
    let mantissa = v / 10f32.powi(exp);
    // Round mantissa to 1 fractional digit; if rounding pushes it to 10,
    // bump the exponent so we still show "1.0e3" not "10.0e2".
    let rounded = (mantissa * 10.0).round() / 10.0;
    let (m, e) = if rounded.abs() >= 10.0 {
        (rounded / 10.0, exp + 1)
    } else {
        (rounded, exp)
    };
    if e == 0 {
        format!("{m:.1}")
    } else {
        format!("{m:.1}e{e}")
    }
}
