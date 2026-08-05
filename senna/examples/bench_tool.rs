// Dense numeric loops where one index addresses several arrays at once; the
// iterator rewrites read worse than the maths they implement.
#![allow(clippy::needless_range_loop)]

//! Benchmark harness for `senna deconvolve`, scored against `data-beans-sim`
//! ground truth. See `senna/docs/deconvolve.md` §8.
//!
//! ```text
//! markers  <dict.parquet> <out.tsv> <topN>
//!     top-N genes per topic → marker TSV
//! score    <true_fractions.parquet> <fractions_ci.tsv> [dict.parquet] [feature_list.parquet]
//!     fraction accuracy + CI calibration; with `dict` the CELL-fraction truth
//!     is converted to the mRNA-fraction scale
//! expr     <expression_dir> <true_dict.parquet> <true_fractions.parquet>
//!     per-sample × per-cell-type expression: profile SHAPE and abundance SCALE
//! profile  <cell_emb> <prop> <rho_dict> <feature_bias> <true_dict> [ln_batch]
//!     how well exp(ρ·t_c + a) reproduces the true per-type profile at the TRUE centroid
//! anchors  <cell_emb> <prop> <feature_emb> <markers.tsv> [posterior_anchors.parquet]
//!     marker/co-embedding anchors vs true cell-type centroids (and ESS drift)
//! ```

use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("markers") => markers(&args[2], &args[3], args[4].parse()?),
        Some("score") => score(
            &args[2],
            &args[3],
            args.get(4).map(String::as_str),
            args.get(5).map(String::as_str),
        ),
        Some("expr") => expr(&args[2], &args[3], &args[4]),
        Some("profile") => profile(
            &args[2],
            &args[3],
            &args[4],
            &args[5],
            &args[6],
            args.get(7).map(String::as_str),
        ),
        Some("anchors") => anchors(
            &args[2],
            &args[3],
            &args[4],
            &args[5],
            args.get(6).map(String::as_str),
        ),
        _ => {
            eprintln!(
                "usage: bench_tool <markers|score|expr|profile|anchors> ... (see module docs)"
            );
            Ok(())
        }
    }
}

fn markers(dict: &str, out: &str, top_n: usize) -> anyhow::Result<()> {
    let m = DMatrix::<f32>::from_parquet(dict)?; // G×K
    let (g, k) = (m.mat.nrows(), m.mat.ncols());
    let mut lines = Vec::new();
    for kk in 0..k {
        let mut idx: Vec<usize> = (0..g).collect();
        idx.sort_by(|&a, &b| m.mat[(b, kk)].total_cmp(&m.mat[(a, kk)]));
        for &gi in idx.iter().take(top_n) {
            lines.push(format!("{}\t{}", m.rows[gi], m.cols[kk]));
        }
    }
    std::fs::write(out, lines.join("\n"))?;
    eprintln!(
        "wrote {} marker rows ({top_n}/topic × {k} topics) → {out}",
        lines.len()
    );
    Ok(())
}

fn score(
    truth: &str,
    ci_tsv: &str,
    dict: Option<&str>,
    feat_list: Option<&str>,
) -> anyhow::Result<()> {
    let mut t = DMatrix::<f32>::from_parquet(truth)?; // rows=sample, cols=celltype
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
    let tf = DMatrix::<f32>::from_parquet(true_frac)?; // S×K true fractions
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
fn profile(
    cell_emb: &str,
    prop: &str,
    rho_dict: &str,
    feat_bias: &str,
    true_dict: &str,
    ln_batch: Option<&str>,
) -> anyhow::Result<()> {
    let ce = DMatrix::<f32>::from_parquet(cell_emb)?;
    let pr = DMatrix::<f32>::from_parquet(prop)?;
    let rho = DMatrix::<f32>::from_parquet(rho_dict)?; // D×H (gated ρ)
    let bias = DMatrix::<f32>::from_parquet(feat_bias)?; // D×1
    let tb = DMatrix::<f32>::from_parquet(true_dict)?; // G×K true beta
    let h = ce.mat.ncols();
    let k = pr.mat.ncols();

    let mut label: HashMap<&str, usize> = HashMap::new();
    for (i, b) in pr.rows.iter().enumerate() {
        let mut best = (0usize, f32::NEG_INFINITY);
        for kk in 0..k {
            if pr.mat[(i, kk)] > best.1 {
                best = (kk, pr.mat[(i, kk)]);
            }
        }
        label.insert(b.as_ref(), best.0);
    }
    let mut tc = vec![vec![0f64; h]; k];
    let mut cnt = vec![0usize; k];
    for (i, b) in ce.rows.iter().enumerate() {
        if let Some(&c) = label.get(b.as_ref()) {
            cnt[c] += 1;
            for j in 0..h {
                tc[c][j] += f64::from(ce.mat[(i, j)]);
            }
        }
    }
    for c in 0..k {
        if cnt[c] > 0 {
            for j in 0..h {
                tc[c][j] /= cnt[c] as f64;
            }
        }
    }

    // genes shared between the reference (ρ rows) and the simulator's β
    let true_row: HashMap<&str, usize> = tb
        .rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.as_ref(), i))
        .collect();
    let pairs: Vec<(usize, usize)> = rho
        .rows
        .iter()
        .enumerate()
        .filter_map(|(di, g)| true_row.get(g.as_ref()).map(|&ti| (di, ti)))
        .collect();
    // Optional control: the simulator's rate carries a per-gene batch factor
    // δ(g) that the reference's `a_g` absorbs; without it the comparison is
    // unfair to the reconstruction.
    let delta: Option<HashMap<String, f64>> = match ln_batch {
        Some(p) => {
            let lb = DMatrix::<f32>::from_parquet(p)?;
            Some(
                lb.rows
                    .iter()
                    .enumerate()
                    .map(|(i, g)| (g.to_string(), f64::from(lb.mat[(i, 0)]).exp()))
                    .collect(),
            )
        }
        None => None,
    };
    println!(
        "genes matched: {} (of {} in reference); delta control: {}",
        pairs.len(),
        rho.rows.len(),
        if delta.is_some() { "ON" } else { "off" }
    );
    // M_c = Σ_g μ_{g,c}: the per-type "exposure". BayesPrism normalizes its
    // reference so this is 1 for every type (making theta an mRNA fraction);
    // ours is unnormalized, so M_c varies and w_c is cell-count-like.
    let mut mc = Vec::new();
    for c in 0..k {
        let mut tot = 0f64;
        for &(di, _) in &pairs {
            let mut s = f64::from(bias.mat[(di, 0)]);
            for j in 0..h {
                s += f64::from(rho.mat[(di, j)]) * tc[c][j];
            }
            tot += s.clamp(-30.0, 30.0).exp();
        }
        mc.push(tot);
    }
    let mean_mc: f64 = mc.iter().sum::<f64>() / mc.len() as f64;
    println!(
        "reference exposure M_c/mean = {:?}  (BayesPrism normalizes these to 1)",
        mc.iter()
            .map(|v| (v / mean_mc * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );

    println!("\n{:<8} {:>14} {:>14}", "type", "corr(raw)", "corr(log)");
    for c in 0..k.min(tb.mat.ncols()) {
        let mut rec = Vec::with_capacity(pairs.len());
        let mut tru = Vec::with_capacity(pairs.len());
        for &(di, ti) in &pairs {
            let mut s = f64::from(bias.mat[(di, 0)]);
            for j in 0..h {
                s += f64::from(rho.mat[(di, j)]) * tc[c][j];
            }
            rec.push(s.clamp(-30.0, 30.0).exp() as f32);
            let d = delta
                .as_ref()
                .and_then(|m| m.get(rho.rows[di].as_ref()).copied())
                .unwrap_or(1.0);
            tru.push((f64::from(tb.mat[(ti, c)]) * d) as f32);
        }
        let lr: Vec<f32> = rec
            .iter()
            .map(|&v| (f64::from(v) + 1e-12).ln() as f32)
            .collect();
        let lt: Vec<f32> = tru
            .iter()
            .map(|&v| (f64::from(v) + 1e-12).ln() as f32)
            .collect();
        println!(
            "{:<8} {:>14.4} {:>14.4}",
            tb.cols.get(c).map_or("?", |s| s.as_ref()),
            pearson(&rec, &tru),
            pearson(&lr, &lt)
        );
    }
    Ok(())
}

/// Compare the marker/co-embedding anchors deconvolve actually uses against the
/// TRUE per-cell-type centroids in the same latent (from labelled cells). Tests
/// whether the co-embedding shrinks anchors toward the global centroid, which
/// would under-separate the reference and bias the fractions systematically.
fn anchors(
    cell_emb: &str,
    prop: &str,
    feat_emb: &str,
    markers_tsv: &str,
    post: Option<&str>,
) -> anyhow::Result<()> {
    let ce = DMatrix::<f32>::from_parquet(cell_emb)?; // N×H (barcodes)
    let pr = DMatrix::<f32>::from_parquet(prop)?; // N×K (barcodes, true theta)
    let fe = DMatrix::<f32>::from_parquet(feat_emb)?; // D×H (genes)
    let h = ce.mat.ncols();

    // true label per barcode = argmax of theta
    let mut label: HashMap<&str, usize> = HashMap::new();
    for (i, b) in pr.rows.iter().enumerate() {
        let mut best = (0usize, f32::NEG_INFINITY);
        for k in 0..pr.mat.ncols() {
            if pr.mat[(i, k)] > best.1 {
                best = (k, pr.mat[(i, k)]);
            }
        }
        label.insert(b.as_ref(), best.0);
    }
    let k = pr.mat.ncols();

    // TRUE centroids: mean cell embedding per label; and the global centroid.
    let mut tc = vec![vec![0f64; h]; k];
    let mut cnt = vec![0usize; k];
    let mut g = vec![0f64; h];
    let mut n_used = 0usize;
    for (i, b) in ce.rows.iter().enumerate() {
        let Some(&c) = label.get(b.as_ref()) else {
            continue;
        };
        cnt[c] += 1;
        n_used += 1;
        for j in 0..h {
            tc[c][j] += f64::from(ce.mat[(i, j)]);
            g[j] += f64::from(ce.mat[(i, j)]);
        }
    }
    for j in 0..h {
        g[j] /= n_used as f64;
    }
    for c in 0..k {
        if cnt[c] > 0 {
            for j in 0..h {
                tc[c][j] /= cnt[c] as f64;
            }
        }
    }

    // CO-EMBEDDING anchors: mean marker-gene co-embedding row per type.
    let gene_row: HashMap<&str, usize> = fe
        .rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.as_ref(), i))
        .collect();
    let mut types: Vec<String> = Vec::new();
    let mut ac: Vec<Vec<f64>> = Vec::new();
    let mut acn: Vec<usize> = Vec::new();
    for line in std::fs::read_to_string(markers_tsv)?.lines() {
        let mut it = line.split('\t');
        let (Some(gene), Some(ty)) = (it.next(), it.next()) else {
            continue;
        };
        let ti = types.iter().position(|t| t == ty).unwrap_or_else(|| {
            types.push(ty.to_string());
            ac.push(vec![0f64; h]);
            acn.push(0);
            types.len() - 1
        });
        if let Some(&gi) = gene_row.get(gene) {
            acn[ti] += 1;
            for j in 0..h {
                ac[ti][j] += f64::from(fe.mat[(gi, j)]);
            }
        }
    }
    for ti in 0..types.len() {
        if acn[ti] > 0 {
            for j in 0..h {
                ac[ti][j] /= acn[ti] as f64;
            }
        }
    }

    let dist = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    };
    let cos = |a: &[f64], b: &[f64]| -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na > 0.0 && nb > 0.0 {
            d / (na * nb)
        } else {
            f64::NAN
        }
    };

    println!(
        "cells used = {n_used}, K = {k}, H = {h}, marker types = {}",
        types.len()
    );
    println!(
        "\n{:<8} {:>12} {:>12} {:>10} {:>10}",
        "type", "|true-g|", "|anchor-g|", "shrink", "cos"
    );
    let m = k.min(types.len());
    for c in 0..m {
        let td: Vec<f64> = (0..h).map(|j| tc[c][j] - g[j]).collect();
        let ad: Vec<f64> = (0..h).map(|j| ac[c][j] - g[j]).collect();
        println!(
            "{:<8} {:>12.4} {:>12.4} {:>10.3} {:>10.3}",
            types.get(c).map_or("?", String::as_str),
            dist(&tc[c], &g),
            dist(&ac[c], &g),
            dist(&ac[c], &g) / dist(&tc[c], &g).max(1e-12),
            cos(&td, &ad)
        );
    }
    // mutual separation: how distinct the references are from each other
    let mut sep_t = 0f64;
    let mut sep_a = 0f64;
    let mut np = 0usize;
    for i in 0..m {
        for j in (i + 1)..m {
            sep_t += dist(&tc[i], &tc[j]);
            sep_a += dist(&ac[i], &ac[j]);
            np += 1;
        }
    }
    println!(
        "\nmean pairwise separation: true = {:.4}, co-embed anchors = {:.4}  ({:.0}% of true)",
        sep_t / np as f64,
        sep_a / np as f64,
        100.0 * (sep_a / np as f64) / (sep_t / np as f64)
    );

    // Did the ESS anchor update move anchors TOWARD the true centroids?
    if let Some(p) = post {
        let pa = DMatrix::<f32>::from_parquet(p)?; // C×H posterior anchors
        println!(
            "\n{:<8} {:>14} {:>14} {:>10}",
            "type", "prior→true", "post→true", "improved?"
        );
        let mut better = 0usize;
        for c in 0..m.min(pa.mat.nrows()) {
            let pv: Vec<f64> = (0..h).map(|j| f64::from(pa.mat[(c, j)])).collect();
            let d_prior = dist(&ac[c], &tc[c]);
            let d_post = dist(&pv, &tc[c]);
            if d_post < d_prior {
                better += 1;
            }
            println!(
                "{:<8} {:>14.4} {:>14.4} {:>10}",
                types.get(c).map_or("?", String::as_str),
                d_prior,
                d_post,
                if d_post < d_prior { "yes" } else { "NO" }
            );
        }
        println!(
            "anchors moved closer to truth: {better}/{}",
            m.min(pa.mat.nrows())
        );
    }
    Ok(())
}

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
