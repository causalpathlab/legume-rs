//! Benchmark harness for `senna deconvolve`, scored against `data-beans-sim`
//! ground truth. See `senna/docs/deconvolve.md`.
//!
//! ```text
//! score    <true_fractions.parquet> <fractions_ci.tsv> [dict.parquet] [feature_list.parquet]
//!     fraction accuracy + CI calibration; with `dict` the CELL-fraction truth
//!     is converted to the mRNA-fraction scale
//! expr     <expression_dir> <true_dict.parquet> <true_fractions.parquet>
//!     per-sample × per-cell-type expression: profile SHAPE and abundance SCALE
//! ```
//!
//! The `markers`, `profile` and `anchors` subcommands are gone with the
//! reconstructed reference they measured. In particular `anchors` read
//! `{out}.anchors.parquet` as one row per cell type; that file now holds one row
//! per archetype, so it would have silently compared unrelated rows.
// Dense numeric loops where one index addresses several arrays at once; the
// iterator rewrites read worse than the maths they implement.
#![allow(clippy::needless_range_loop)]

use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("score") => score(
            &args[2],
            &args[3],
            args.get(4).map(String::as_str),
            args.get(5).map(String::as_str),
        ),
        Some("shape") => {
            for p in &args[2..] {
                match DMatrix::<f32>::from_parquet(p) {
                    Ok(m) => {
                        let v: Vec<f32> =
                            (0..m.mat.ncols().min(3)).map(|j| m.mat[(0, j)]).collect();
                        println!(
                            "{:<44} {:>6} x {:<4} rows[{}..] cols[{}..] head={:?}",
                            p.rsplit('/').next().unwrap_or(p),
                            m.mat.nrows(),
                            m.mat.ncols(),
                            m.rows.first().map_or("", |s| s.as_ref()),
                            m.cols.first().map_or("", |s| s.as_ref()),
                            v
                        );
                    }
                    Err(e) => println!("{p}: <unreadable: {e}>"),
                }
            }
            Ok(())
        }
        // TSV (with header) -> parquet, preserving row/column names. `senna`'s
        // `read_mat` passes header_row=None, so a headered TSV cannot be fed
        // directly to --bulk.
        Some("topar") => {
            let m = load_named(&args[2])?;
            m.mat
                .to_parquet_with_names(&args[3], (Some(&m.rows), Some("gene")), Some(&m.cols))?;
            eprintln!("wrote {} x {} -> {}", m.mat.nrows(), m.mat.ncols(), args[3]);
            Ok(())
        }
        Some("expr") => expr(&args[2], &args[3], &args[4]),
        _ => {
            eprintln!("usage: bench_tool <score|expr|shape|topar> ... (see module docs)");
            Ok(())
        }
    }
}

/// Load a named matrix from parquet, or from a delimited text file (header row,
/// row names in column 0) so ground truth can be supplied either way.
fn load_named(
    path: &str,
) -> anyhow::Result<matrix_util::traits::MatWithNames<nalgebra::DMatrix<f32>>> {
    if path.ends_with(".parquet") {
        DMatrix::<f32>::from_parquet(path)
    } else {
        DMatrix::<f32>::read_data(path, &['\t', ','][..], Some(0), Some(0), None, None)
    }
}

fn score(
    truth: &str,
    ci_tsv: &str,
    dict: Option<&str>,
    feat_list: Option<&str>,
) -> anyhow::Result<()> {
    let mut t = load_named(truth)?; // rows=sample, cols=celltype
                                    // Optional: convert the CELL-fraction truth into the mRNA-fraction the model
                                    // actually estimates. Per-type output s_k = Σ_g β(g,k) (restricted to the
                                    // genes the reference retained, since only those carry counts here).
    if let Some(dpath) = dict {
        let d = DMatrix::<f32>::from_parquet(dpath)?; // G×K
        let keep: Option<HashMap<String, ()>> = match feat_list {
            Some(f) => Some(
                DMatrix::<f32>::from_parquet(f)?
                    .rows
                    .into_iter()
                    .map(|r| (r.to_string(), ()))
                    .collect(),
            ),
            None => None,
        };
        let mut s: HashMap<String, f64> = HashMap::new();
        for (ci, cname) in d.cols.iter().enumerate() {
            let mut tot = 0f64;
            for (gi, gname) in d.rows.iter().enumerate() {
                if keep
                    .as_ref()
                    .is_some_and(|k| !k.contains_key(gname.as_ref()))
                {
                    continue;
                }
                tot += f64::from(d.mat[(gi, ci)]);
            }
            s.insert(cname.to_string(), tot);
        }
        let sk: Vec<f64> = t
            .cols
            .iter()
            .map(|c| *s.get(c.as_ref()).unwrap_or(&1.0))
            .collect();
        let mean_s: f64 = sk.iter().sum::<f64>() / sk.len() as f64;
        eprintln!(
            "converted truth to mRNA-fractions; relative per-type output s_k/mean = {:?}",
            sk.iter()
                .map(|v| (v / mean_s * 100.0).round() / 100.0)
                .collect::<Vec<_>>()
        );
        for si in 0..t.mat.nrows() {
            let denom: f64 = (0..t.mat.ncols())
                .map(|k| f64::from(t.mat[(si, k)]) * sk[k])
                .sum();
            for k in 0..t.mat.ncols() {
                t.mat[(si, k)] = (f64::from(t.mat[(si, k)]) * sk[k] / denom.max(1e-12)) as f32;
            }
        }
    }
    // (sample,celltype) -> (mean, sd, lo, hi)
    let mut est: HashMap<(String, String), (f32, f32, f32, f32)> = HashMap::new();
    for (i, line) in std::fs::read_to_string(ci_tsv)?.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 6 {
            continue;
        }
        est.insert(
            (f[0].to_string(), f[1].to_string()),
            (f[2].parse()?, f[3].parse()?, f[4].parse()?, f[5].parse()?),
        );
    }

    let mut xs = Vec::new(); // true
    let mut ys = Vec::new(); // est mean
    let mut covered = 0usize;
    let mut n = 0usize;
    let mut abs_sum = 0f64;
    let mut sq_sum = 0f64;
    let mut per_ct: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();
    let mut missing = 0usize;
    // Calibration bookkeeping: posterior spread vs. actual error, and the
    // per-cell-type signed error (the systematic component).
    let mut sd_sum = 0f64;
    let mut width_sum = 0f64;
    let mut z_sum = 0f64;
    let mut signed_by_ct: HashMap<String, (f64, usize)> = HashMap::new();
    for (si, sname) in t.rows.iter().enumerate() {
        for (ci, cname) in t.cols.iter().enumerate() {
            let truth_v = t.mat[(si, ci)];
            let Some(&(mean, sd, lo, hi)) = est.get(&(sname.to_string(), cname.to_string())) else {
                missing += 1;
                continue;
            };
            xs.push(truth_v);
            ys.push(mean);
            let e = per_ct.entry(cname.to_string()).or_default();
            e.0.push(truth_v);
            e.1.push(mean);
            if truth_v >= lo && truth_v <= hi {
                covered += 1;
            }
            let err = f64::from(mean - truth_v);
            abs_sum += err.abs();
            sq_sum += err.powi(2);
            sd_sum += f64::from(sd);
            width_sum += f64::from(hi - lo);
            if sd > 0.0 {
                z_sum += (err / f64::from(sd)).abs();
            }
            let b = signed_by_ct.entry(cname.to_string()).or_insert((0.0, 0));
            b.0 += err;
            b.1 += 1;
            n += 1;
        }
    }
    if missing > 0 {
        eprintln!("WARNING: {missing} (sample,celltype) pairs had no estimate (name mismatch?)");
        eprintln!("  truth samples e.g.: {:?}", &t.rows[..t.rows.len().min(2)]);
        eprintln!("  truth celltypes: {:?}", t.cols);
        let ks: Vec<_> = est.keys().take(2).cloned().collect();
        eprintln!("  est keys e.g.: {ks:?}");
    }

    println!(
        "n pairs        = {n}   ({} samples × {} celltypes)",
        t.rows.len(),
        t.cols.len()
    );
    println!("overall Pearson= {:.4}", pearson(&xs, &ys));
    println!("RMSE           = {:.4}", (sq_sum / n as f64).sqrt());
    println!("MAE            = {:.4}", abs_sum / n as f64);
    println!(
        "95% CI coverage= {:.1}%  ({covered}/{n})",
        100.0 * covered as f64 / n as f64
    );

    // --- why coverage misses nominal -------------------------------------
    // A calibrated 95% interval needs posterior sd ≈ actual error. Compare
    // them, then split the squared error into a systematic per-cell-type
    // shift (bias, which no symmetric interval around a shifted centre can
    // cover) and the remainder.
    let nf = n as f64;
    let rmse = (sq_sum / nf).sqrt();
    let mean_sd = sd_sum / nf;
    let mse_bias: f64 = signed_by_ct
        .values()
        .map(|&(s, c)| (s / c as f64).powi(2) * c as f64)
        .sum::<f64>()
        / nf;
    println!("--- calibration ---");
    println!("mean posterior sd   = {mean_sd:.5}");
    println!("mean 95% CI width   = {:.5}", width_sum / nf);
    println!(
        "RMSE / mean sd      = {:.1}x   (1x = calibrated)",
        rmse / mean_sd
    );
    println!(
        "mean |z|            = {:.1}    (0.8 if calibrated)",
        z_sum / nf
    );
    println!(
        "bias share of MSE   = {:.0}%  (systematic per-celltype shift)",
        100.0 * mse_bias / (sq_sum / nf)
    );
    let mut cts: Vec<_> = per_ct.keys().cloned().collect();
    cts.sort();
    for ct in cts {
        let (x, y) = &per_ct[&ct];
        println!(
            "  {ct:<10} per-type Pearson = {:.4}  (mean true {:.3}, mean est {:.3})",
            pearson(x, y),
            x.iter().sum::<f32>() / x.len() as f32,
            y.iter().sum::<f32>() / y.len() as f32
        );
    }
    Ok(())
}

/// Accuracy of the per-sample × per-cell-type expression tensor `E[Z]`.
/// Two things are checked separately:
///   (a) SHAPE  — does `E[Z_{s,c,·}]` recover cell type c's gene profile?
///   (b) SCALE  — does `Σ_g E[Z_{s,c,g}]` track that type's true abundance?
fn expr(expr_dir: &str, true_dict: &str, true_frac: &str) -> anyhow::Result<()> {
    let tb = DMatrix::<f32>::from_parquet(true_dict)?; // G×K true beta
    let tf = load_named(true_frac)?; // S×K true fractions
    println!(
        "{:<8} {:>16} {:>16} {:>14}",
        "type", "shape corr(raw)", "shape corr(log)", "scale corr"
    );
    let (mut sr, mut sl, mut sc_, mut ntyp) = (0f64, 0f64, 0f64, 0usize);
    for (ci, cname) in tb.cols.iter().enumerate() {
        let path = format!("{expr_dir}/{cname}.parquet");
        let Ok(z) = DMatrix::<f32>::from_parquet(&path) else {
            eprintln!("skip {path} (not found)");
            continue;
        };
        // gene alignment: z cols are genes, tb rows are genes
        let trow: HashMap<&str, usize> = tb
            .rows
            .iter()
            .enumerate()
            .map(|(i, g)| (g.as_ref(), i))
            .collect();
        let pairs: Vec<(usize, usize)> = z
            .cols
            .iter()
            .enumerate()
            .filter_map(|(zi, g)| trow.get(g.as_ref()).map(|&ti| (zi, ti)))
            .collect();
        // (a) per-sample profile recovery, averaged over samples
        let (mut craw, mut clog, mut ns) = (0f64, 0f64, 0usize);
        let mut totals = Vec::new();
        for si in 0..z.mat.nrows() {
            let mut a = Vec::with_capacity(pairs.len());
            let mut b = Vec::with_capacity(pairs.len());
            for &(zi, ti) in &pairs {
                a.push(z.mat[(si, zi)]);
                b.push(tb.mat[(ti, ci)]);
            }
            let la: Vec<f32> = a
                .iter()
                .map(|&v| (f64::from(v) + 1.0).ln() as f32)
                .collect();
            let lb: Vec<f32> = b
                .iter()
                .map(|&v| (f64::from(v) + 1e-12).ln() as f32)
                .collect();
            craw += pearson(&a, &b);
            clog += pearson(&la, &lb);
            ns += 1;
            totals.push(a.iter().map(|&v| f64::from(v)).sum::<f64>() as f32);
        }
        // (b) does the assigned total track the true abundance across samples?
        let truth_col: Vec<f32> = (0..tf.mat.nrows().min(totals.len()))
            .map(|si| tf.mat[(si, ci)])
            .collect();
        let scale = pearson(&totals[..truth_col.len()], &truth_col);
        println!(
            "{:<8} {:>16.4} {:>16.4} {:>14.4}",
            cname.as_ref(),
            craw / ns as f64,
            clog / ns as f64,
            scale
        );
        sr += craw / ns as f64;
        sl += clog / ns as f64;
        sc_ += scale;
        ntyp += 1;
    }
    if ntyp > 0 {
        println!(
            "\nmean over types: shape(raw)={:.4}  shape(log)={:.4}  scale={:.4}",
            sr / ntyp as f64,
            sl / ntyp as f64,
            sc_ / ntyp as f64
        );
    }
    Ok(())
}

/// How well does the embedding reconstruction `μ_c = exp(ρ·t_c + a)` reproduce the
/// TRUE per-cell-type gene profile, when `t_c` is the true cell-type centroid?
/// This isolates the reference model from anchor placement.
fn pearson(x: &[f32], y: &[f32]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let (sx, sy): (f64, f64) = (
        x.iter().map(|&v| v as f64).sum(),
        y.iter().map(|&v| v as f64).sum(),
    );
    let sxx: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum();
    let syy: f64 = y.iter().map(|&v| (v as f64).powi(2)).sum();
    let sxy: f64 = x.iter().zip(y).map(|(&a, &b)| a as f64 * b as f64).sum();
    let cov = sxy - sx * sy / n;
    let vx = sxx - sx * sx / n;
    let vy = syy - sy * sy / n;
    if vx > 0.0 && vy > 0.0 {
        cov / (vx.sqrt() * vy.sqrt())
    } else {
        f64::NAN
    }
}
