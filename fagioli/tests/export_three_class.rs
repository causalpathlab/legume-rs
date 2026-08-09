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
use three_class::{simulate_classes, NUM_BLOCKS, NUM_TRAITS, SNPS_PER_BLOCK};

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
        println!("wrote seed {seed}: {m} variants, {NUM_TRAITS} traits, {NUM_BLOCKS} blocks");
    }

    println!(
        "\n{} individuals per cohort; blocks are {} SNPs each.\nOutput in {OUT}/",
        three_class::NUM_INDIVIDUALS,
        SNPS_PER_BLOCK,
    );
    Ok(())
}
