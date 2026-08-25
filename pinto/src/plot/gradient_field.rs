//! Per-cell gradient field over LR contacts, and field lines through it.
//!
//! The per-edge quiver draws one arrow per contact, thousands of glyphs
//! radiating in every direction around each cell. Averaging those edge
//! segments directly cancels to nothing: a hub's contacts point outward
//! all around it by construction. What has spatial structure is the
//! per-CELL resultant, one net ligand-to-receptor direction per cell,
//! accumulated over the cell's contacts. Those resultants are binned,
//! kernel smoothed, and either drawn one arrow per bin or integrated
//! into field lines with evenly spaced seeds.
//!
//! WHAT THE FIELD IS, and is not. A cell's vector is the direction of
//! its local ligand-to-receptor EXPRESSION GRADIENT: toward the
//! neighbors its drawn contacts point at. It is not a velocity, and the
//! co-activity estimand it illustrates is symmetric in the pair, so
//! nothing here may be read as signalling propagating across the
//! tissue. The vocabulary is deliberately static: these are field lines
//! of a gradient, in the same sense as field lines of a potential,
//! never flow.
//!
//! COHERENCE is the honesty gate. Each bin records the resultant length
//! of its (smoothed) unit vectors, 1 when the cells agree and ~0 when
//! they cancel. Field lines neither seed in nor continue into bins
//! below the floor: where the data has no direction, the figure draws
//! none, and smoothing never blurs direction into ground that had
//! nothing to say (the blur is count-weighted, so empty bins borrow
//! only what their neighbours measured).

use plot_utils::hull::Pt;
use plot_utils::rasterize::{Extent, Segment};

/// A binned, optionally smoothed vector field in PIXEL coordinates.
///
/// Internally everything stays in accumulator form (sums + counts) so
/// smoothing is one box blur over all planes at once; the public
/// accessors derive means on the fly.
pub struct GradientField {
    pub nx: usize,
    pub ny: usize,
    pub cell_w: f32,
    pub cell_h: f32,
    /// Summed unit vectors per bin.
    sx: Vec<f32>,
    sy: Vec<f32>,
    /// Summed FINITE values and their count (a colour mode may emit NaN).
    sval: Vec<f32>,
    val_n: Vec<f32>,
    /// Points binned here (fractional after smoothing).
    count: Vec<f32>,
}

impl GradientField {
    #[inline]
    fn idx(&self, ix: usize, iy: usize) -> usize {
        iy * self.nx + ix
    }

    fn locate(&self, p: Pt) -> Option<(usize, usize)> {
        if !p.0.is_finite() || !p.1.is_finite() || p.0 < 0.0 || p.1 < 0.0 {
            return None;
        }
        let ix = (p.0 / self.cell_w) as usize;
        let iy = (p.1 / self.cell_h) as usize;
        (ix < self.nx && iy < self.ny).then_some((ix, iy))
    }

    /// `(mean unit direction, coherence, mean value, weight)` at a pixel
    /// point, `None` outside the frame or in a bin that measured nothing.
    /// Coherence is the resultant length `|sum u| / n` in `[0, 1]`; the
    /// value is NaN when no finite value ever arrived (a value-free bin
    /// must not masquerade as a measured neutral one).
    pub fn sample(&self, p: Pt) -> Option<((f32, f32), f32, f32, f32)> {
        let (ix, iy) = self.locate(p)?;
        let k = self.idx(ix, iy);
        let c = self.count[k];
        if c <= 0.0 {
            return None;
        }
        let coh = (self.sx[k] * self.sx[k] + self.sy[k] * self.sy[k]).sqrt() / c;
        let val = if self.val_n[k] > 0.0 {
            self.sval[k] / self.val_n[k]
        } else {
            f32::NAN
        };
        Some(((self.sx[k] / c, self.sy[k] / c), coh, val, c))
    }

    /// Count-weighted 3x3 box blur over every accumulator plane,
    /// `passes` times. Blurring the SUMS keeps the coherence honest: a
    /// bin that measured nothing gains only its neighbours' vectors and
    /// their counts together, so agreement still has to be earned.
    pub fn smooth(&mut self, passes: usize) {
        let (nx, ny) = (self.nx, self.ny);
        for _ in 0..passes {
            let mut planes = [
                vec![0.0f32; nx * ny], // sx
                vec![0.0f32; nx * ny], // sy
                vec![0.0f32; nx * ny], // sval
                vec![0.0f32; nx * ny], // val_n
                vec![0.0f32; nx * ny], // count
            ];
            for iy in 0..ny {
                for ix in 0..nx {
                    let k = iy * nx + ix;
                    let mut acc = [0.0f32; 5];
                    for dy in -1i64..=1 {
                        for dx in -1i64..=1 {
                            let (jx, jy) = (ix as i64 + dx, iy as i64 + dy);
                            if jx < 0 || jy < 0 || jx >= nx as i64 || jy >= ny as i64 {
                                continue;
                            }
                            let j = jy as usize * nx + jx as usize;
                            // Centre weighted 2x so structure is kept.
                            let w = if dx == 0 && dy == 0 { 2.0 } else { 1.0 };
                            acc[0] += w * self.sx[j];
                            acc[1] += w * self.sy[j];
                            acc[2] += w * self.sval[j];
                            acc[3] += w * self.val_n[j];
                            acc[4] += w * self.count[j];
                        }
                    }
                    for (plane, a) in planes.iter_mut().zip(acc) {
                        plane[k] = a * 0.1;
                    }
                }
            }
            let [sx, sy, sval, val_n, count] = planes;
            self.sx = sx;
            self.sy = sy;
            self.sval = sval;
            self.val_n = val_n;
            self.count = count;
        }
    }

    /// Bilinearly interpolated `(direction, coherence, value, weight)`
    /// at a pixel point, over the four surrounding bin centres. This is
    /// what the field-line integrator samples; nearest-bin sampling
    /// gives visibly kinked polylines.
    pub fn sample_bilinear(&self, p: Pt) -> Option<((f32, f32), f32, f32, f32)> {
        if !p.0.is_finite() || !p.1.is_finite() {
            return None;
        }
        let gx = (p.0 / self.cell_w - 0.5).clamp(0.0, (self.nx - 1) as f32);
        let gy = (p.1 / self.cell_h - 0.5).clamp(0.0, (self.ny - 1) as f32);
        let (i0, j0) = (gx as usize, gy as usize);
        let (i1, j1) = ((i0 + 1).min(self.nx - 1), (j0 + 1).min(self.ny - 1));
        let (tx, ty) = (gx - i0 as f32, gy - j0 as f32);
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut sval = 0.0;
        let mut val_n = 0.0;
        let mut count = 0.0;
        for (i, j, w) in [
            (i0, j0, (1.0 - tx) * (1.0 - ty)),
            (i1, j0, tx * (1.0 - ty)),
            (i0, j1, (1.0 - tx) * ty),
            (i1, j1, tx * ty),
        ] {
            let k = self.idx(i, j);
            sx += w * self.sx[k];
            sy += w * self.sy[k];
            sval += w * self.sval[k];
            val_n += w * self.val_n[k];
            count += w * self.count[k];
        }
        if count <= 0.0 {
            return None;
        }
        let coh = (sx * sx + sy * sy).sqrt() / count;
        let val = if val_n > 0.0 { sval / val_n } else { f32::NAN };
        Some(((sx / count, sy / count), coh, val, count))
    }

    /// One arrow per bin clearing the floors, scaled by coherence: a bin
    /// whose cells disagree reads as a stub rather than a claim.
    pub fn gridded_arrows(&self, min_coherence: f32, min_count: f32) -> (Vec<Segment>, Vec<f32>) {
        let mut segs = Vec::new();
        let mut vals = Vec::new();
        let full = 0.5 * self.cell_w.min(self.cell_h);
        for iy in 0..self.ny {
            for ix in 0..self.nx {
                let cx = (ix as f32 + 0.5) * self.cell_w;
                let cy = (iy as f32 + 0.5) * self.cell_h;
                let Some(((dx, dy), coh, val, c)) = self.sample((cx, cy)) else {
                    continue;
                };
                if c < min_count || coh < min_coherence {
                    continue;
                }
                let norm = (dx * dx + dy * dy).sqrt();
                if norm <= 0.0 {
                    continue;
                }
                let len = full * coh;
                let (ux, uy) = (dx / norm, dy / norm);
                segs.push((
                    (cx - ux * len, cy - uy * len),
                    (cx + ux * len, cy + uy * len),
                ));
                vals.push(val);
            }
        }
        (segs, vals)
    }
}

/// Build the field from per-POINT vectors (one point per cell, its net
/// contact direction). Vector lengths are normalized away: the field
/// measures agreement in direction, not magnitude.
pub fn build_gradient_field(
    points: &[Pt],
    vectors: &[(f32, f32)],
    values: &[f32],
    ext: Extent,
    target_bins: usize,
) -> GradientField {
    let target_bins = target_bins.clamp(4, 400);
    let (w, h) = (ext.w as f32, ext.h as f32);
    let (nx, ny) = if w >= h {
        let nx = target_bins;
        let ny = ((h / w) * target_bins as f32).round().max(1.0) as usize;
        (nx, ny)
    } else {
        let ny = target_bins;
        let nx = ((w / h) * target_bins as f32).round().max(1.0) as usize;
        (nx, ny)
    };

    let n = nx * ny;
    let mut f = GradientField {
        nx,
        ny,
        cell_w: w / nx as f32,
        cell_h: h / ny as f32,
        sx: vec![0.0; n],
        sy: vec![0.0; n],
        sval: vec![0.0; n],
        val_n: vec![0.0; n],
        count: vec![0.0; n],
    };

    for ((p, v), val) in points.iter().zip(vectors).zip(values) {
        let norm = (v.0 * v.0 + v.1 * v.1).sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            continue;
        }
        let Some((ix, iy)) = f.locate(*p) else {
            continue;
        };
        let k = f.idx(ix, iy);
        f.sx[k] += v.0 / norm;
        f.sy[k] += v.1 / norm;
        if val.is_finite() {
            f.sval[k] += *val;
            f.val_n[k] += 1.0;
        }
        f.count[k] += 1.0;
    }
    f
}

/// Field-line rendering parameters, matching the matplotlib
/// `streamplot` algorithm scVelo draws with: an occupancy mask of
/// `~30 * density` cells per axis spaces the lines (one line per mask
/// cell), lines shorter than `min_length` are discarded and their mask
/// cells released, and integration is adaptive RK12 against the
/// bilinearly interpolated field. Lengths are in NORMALIZED axes units
/// (the frame's longer side is 1), as in matplotlib.
pub struct FieldLineArgs {
    /// Mask resolution scale; 1.0 gives the classic 30 x 30.
    pub density: f32,
    /// Discard lines shorter than this (axes units).
    pub min_length: f32,
    /// Stop extending a line beyond this arc length (axes units).
    pub max_length: f32,
    /// Bins below this coherence neither seed nor continue a line.
    pub min_coherence: f32,
    /// Bins with less weight than this are treated as empty.
    pub min_count: f32,
}

impl Default for FieldLineArgs {
    fn default() -> Self {
        Self {
            density: 1.5,
            min_length: 0.05,
            max_length: 4.0,
            min_coherence: 0.12,
            min_count: 0.4,
        }
    }
}

/// Field lines as whole polylines (PIXEL coordinates), one mean field
/// value per line. One arrowhead belongs at each line's mid-arc point;
/// the caller renders that, keeping heads one-per-line like streamplot.
pub struct FieldLines {
    pub lines: Vec<Vec<Pt>>,
    pub values: Vec<f32>,
}

/// Integrate field lines through the field, the streamplot way.
///
/// Seeds walk the mask in raster order; each trajectory runs backward
/// then forward with adaptive RK12 at unit speed (steps bounded by one
/// mask cell so no cell is skipped), claims every mask cell it enters,
/// stops on collision with a claimed cell, and is discarded again —
/// mask cells released — when its total arc length ends up below
/// `min_length`. The coherence and weight floors terminate integration
/// exactly like leaving the grid: where the cells disagree, there is no
/// line to draw.
pub fn field_lines(field: &GradientField, args: &FieldLineArgs) -> FieldLines {
    let (w, h) = (
        field.nx as f32 * field.cell_w,
        field.ny as f32 * field.cell_h,
    );
    let long = w.max(h);
    // Normalized axes coordinates: pixel / long side.
    let mask_nx = ((30.0 * args.density * w / long).round() as usize).max(4);
    let mask_ny = ((30.0 * args.density * h / long).round() as usize).max(4);
    let (mcw, mch) = (w / mask_nx as f32, h / mask_ny as f32);
    let mut mask = vec![false; mask_nx * mask_ny];
    let mask_at = |p: Pt| -> Option<usize> {
        let ix = (p.0 / mcw) as usize;
        let iy = (p.1 / mch) as usize;
        (p.0 >= 0.0 && p.1 >= 0.0 && ix < mask_nx && iy < mask_ny).then(|| iy * mask_nx + ix)
    };

    // Unit-speed direction, or None where the field ends / fails floors.
    let dir = |p: Pt| -> Option<(f32, f32)> {
        let ((dx, dy), coh, _, c) = field.sample_bilinear(p)?;
        (coh >= args.min_coherence && c >= args.min_count).then(|| {
            let n = (dx * dx + dy * dy).sqrt().max(1e-12);
            (dx / n, dy / n)
        })
    };

    let maxerror = 0.003f32;
    // Step bound: one mask cell, so no cell on the path is skipped.
    let maxds = (mcw.min(mch) / long).min(0.1);

    let mut out = FieldLines {
        lines: Vec::new(),
        values: Vec::new(),
    };

    for my in 0..mask_ny {
        for mx in 0..mask_nx {
            if mask[my * mask_nx + mx] {
                continue;
            }
            let seed = ((mx as f32 + 0.5) * mcw, (my as f32 + 0.5) * mch);
            if dir(seed).is_none() {
                continue;
            }

            let mut claimed: Vec<usize> = Vec::new();
            let claim = |p: Pt, claimed: &mut Vec<usize>, mask: &mut Vec<bool>| -> bool {
                match mask_at(p) {
                    Some(k) => {
                        if mask[k] && !claimed.contains(&k) {
                            return false; // collision with another line
                        }
                        if !mask[k] {
                            mask[k] = true;
                            claimed.push(k);
                        }
                        true
                    }
                    None => false,
                }
            };
            claim(seed, &mut claimed, &mut mask);

            let mut total_len = 0.0f32;
            let mut halves: [Vec<Pt>; 2] = [vec![seed], vec![seed]];
            for (half, sign) in halves.iter_mut().zip([-1.0f32, 1.0]) {
                let mut p = seed;
                let mut ds = maxds;
                let mut len = 0.0f32;
                while len < args.max_length {
                    let Some(d1) = dir(p) else { break };
                    let q1 = (p.0 + sign * d1.0 * ds * long, p.1 + sign * d1.1 * ds * long);
                    let Some(d2) = dir(q1) else {
                        // Shrink toward the boundary rather than stopping
                        // a full step short of it.
                        if ds > maxds * 0.1 {
                            ds *= 0.5;
                            continue;
                        }
                        break;
                    };
                    let err = 0.5 * ds * ((d2.0 - d1.0).powi(2) + (d2.1 - d1.1).powi(2)).sqrt();
                    if err > maxerror && ds > maxds * 0.1 {
                        ds *= 0.5;
                        continue;
                    }
                    let q = (
                        p.0 + sign * (d1.0 + d2.0) * 0.5 * ds * long,
                        p.1 + sign * (d1.1 + d2.1) * 0.5 * ds * long,
                    );
                    if dir(q).is_none() || !claim(q, &mut claimed, &mut mask) {
                        break;
                    }
                    half.push(q);
                    p = q;
                    len += ds;
                    // Grow the step back toward the cap when accuracy allows.
                    ds = (ds * 1.85).min(maxds);
                }
                total_len += len;
            }

            if total_len < args.min_length {
                for k in claimed {
                    mask[k] = false;
                }
                continue;
            }
            // Backward half reversed, then forward: one continuous line.
            let [back, fwd] = halves;
            let mut line: Vec<Pt> = back.into_iter().rev().collect();
            line.extend(fwd.into_iter().skip(1));
            if line.len() < 2 {
                continue;
            }
            let mut sval = 0.0f32;
            let mut nval = 0.0f32;
            for p in &line {
                if let Some((_, _, v, _)) = field.sample_bilinear(*p) {
                    if v.is_finite() {
                        sval += v;
                        nval += 1.0;
                    }
                }
            }
            out.values
                .push(if nval > 0.0 { sval / nval } else { f32::NAN });
            out.lines.push(line);
        }
    }
    out
}
