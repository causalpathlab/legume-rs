//! Dump the three-class fixture so an external method can be run against it.
//!
//! `#[ignore]` by default: this writes files and exists to feed the R
//! comparison in `analysis/susie_vs_shape.R`, not to check anything itself.
//!
//! Run: cargo test -p fagioli --release --test export_three_class -- --ignored
use anyhow::Result;
use matrix_util::traits::IoOps;
use nalgebra::DMatrix;

#[path = "common/three_class.rs"]
mod three_class;
#[path = "common/pipeline.rs"]
mod pipeline;
use fagioli::embedding::score::assemble_u;
use three_class::{simulate_classes, NUM_BLOCKS, NUM_PROGRAMS, NUM_TRAITS, SNPS_PER_BLOCK};

/// Effective number of nonzero coordinates.
fn participation_ratio(x: impl Iterator<Item = f32>) -> f32 {
    let (s2, s4) = x.fold((0.0f64, 0.0f64), |(a, b), v| {
        let v2 = (v as f64) * (v as f64);
        (a + v2, b + v2 * v2)
    });
    if s4 <= 0.0 {
        return 0.0;
    }
    (s2 * s2 / s4) as f32
}

const OUT: &str = "target/three_class_export";

#[test]
#[ignore = "writes fixture files for the external R comparison"]
fn export_for_external_comparison() -> Result<()> {
    std::fs::create_dir_all(OUT)?;
    let seeds = [20250808u64, 111, 222];

    for seed in seeds {
        let cl = simulate_classes(0.4, seed);
        let m = cl.input.zscores.nrows();

        // z-scores, one row per variant.
        let trait_names: Vec<Box<str>> =
            (0..NUM_TRAITS).map(|t| Box::from(format!("trait_{t}"))).collect();
        let snp_names: Vec<Box<str>> =
            (0..m).map(|g| Box::from(format!("rs{g}"))).collect();
        cl.input.zscores.to_parquet_with_names(
            &format!("{OUT}/z_{seed}.parquet"),
            (Some(&snp_names), Some("snp_id")),
            Some(&trait_names),
        )?;

        // Class labels: 0 null, 1 pleiotropic, 2 trait-specific.
        let labels = DMatrix::from_fn(m, 1, |g, _| {
            if cl.pleiotropic.contains(&g) {
                1.0f32
            } else if cl.trait_specific.contains(&g) {
                2.0
            } else {
                0.0
            }
        });
        labels.to_parquet_with_names(
            &format!("{OUT}/labels_{seed}.parquet"),
            (Some(&snp_names), Some("snp_id")),
            Some(&[Box::from("class")]),
        )?;

        // In-sample LD per block — the favourable case for a method that needs
        // R, since there is no panel mismatch to pay for.
        let x = &cl.input.geno.genotypes;
        let n = x.nrows() as f32;
        for b in 0..NUM_BLOCKS {
            let s = b * SNPS_PER_BLOCK;
            let mut xb = x.columns(s, SNPS_PER_BLOCK).clone_owned();
            {
                use matrix_util::traits::MatOps;
                xb.scale_columns_inplace();
            }
            let r = (xb.transpose() * &xb) / n;
            let names: Vec<Box<str>> = (0..SNPS_PER_BLOCK)
                .map(|j| Box::from(format!("rs{}", s + j)))
                .collect();
            r.to_parquet_with_names(
                &format!("{OUT}/ld_{seed}_block{b}.parquet"),
                (Some(&names), Some("snp_id")),
                Some(&names),
            )?;
        }
        // The NCE arm's per-variant statistics, so R can curve it against
        // susieR without reimplementing the embedding.
        let (fit, starts) = pipeline::fit_embedding(&cl.input, NUM_PROGRAMS, seed)?;
        let u = assemble_u(&fit.u_mean, &starts, m);
        let b_hat = &u * fit.v_check.transpose();
        let nce = DMatrix::from_fn(m, 2, |g, c| match c {
            0 => u.row(g).norm(),
            _ => participation_ratio(b_hat.row(g).iter().copied()),
        });
        nce.to_parquet_with_names(
            &format!("{OUT}/nce_{seed}.parquet"),
            (Some(&snp_names), Some("snp_id")),
            Some(&[Box::from("u_norm"), Box::from("pr_fitted")]),
        )?;

        println!("wrote seed {seed}: {m} variants, {NUM_TRAITS} traits, {NUM_BLOCKS} blocks");
    }

    println!(
        "\n{} individuals per cohort; blocks are {} SNPs each.\nOutput in {OUT}/",
        three_class::NUM_INDIVIDUALS,
        SNPS_PER_BLOCK,
    );
    Ok(())
}
