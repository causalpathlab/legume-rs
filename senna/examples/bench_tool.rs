//! Throwaway benchmark helper for `senna deconvolve`:
//!   markers <dict.parquet> <out.tsv> <topN>   — top-N genes per topic → marker TSV
//!   score   <true_fractions.parquet> <fractions_ci.tsv>  — corr / RMSE / CI coverage

use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("markers") => markers(&args[2], &args[3], args[4].parse()?),
        Some("score") => score(&args[2], &args[3]),
        _ => {
            eprintln!("usage: bench_tool markers <dict.parquet> <out.tsv> <topN> | score <true.parquet> <ci.tsv>");
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

fn score(truth: &str, ci_tsv: &str) -> anyhow::Result<()> {
    let t = DMatrix::<f32>::from_parquet(truth)?; // rows=sample, cols=celltype
                                                  // (sample,celltype) -> (mean, lo, hi)
    let mut est: HashMap<(String, String), (f32, f32, f32)> = HashMap::new();
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
            (f[2].parse()?, f[4].parse()?, f[5].parse()?),
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
    for (si, sname) in t.rows.iter().enumerate() {
        for (ci, cname) in t.cols.iter().enumerate() {
            let truth_v = t.mat[(si, ci)];
            let Some(&(mean, lo, hi)) = est.get(&(sname.to_string(), cname.to_string())) else {
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
            abs_sum += f64::from((mean - truth_v).abs());
            sq_sum += f64::from((mean - truth_v).powi(2));
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
