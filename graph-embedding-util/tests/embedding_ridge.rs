//! Pins the ridge's REDUCTION, which is the only thing about it that matters.
//!
//! `λ · mean_g ‖e_g‖²` was for a long time written `λ · mean_all(E²)`, dividing
//! by `rows · H` instead of `rows`. That made the per-element gradient `H×`
//! smaller than intended — on realistic tables a pull of order `1e-6·e`, which
//! `senna bge` measured as indistinguishable from λ = 0 and then documented as
//! a reason to leave the knob off. Nothing in the workspace tested it, so the
//! dilution survived in two independent copies.
//!
//! These tests fail if either property is lost: independence from the row
//! count (so λ does not move with gene/cell count) and DEPENDENCE on the row
//! norm rather than the per-element mean (so λ does not move with `-d`).

use candle_util::candle_core::{DType, Device, Tensor};
use graph_embedding_util::loss::embedding_ridge;

fn cpu() -> Device {
    Device::Cpu
}

/// A table of all-`c` entries has `‖e_g‖² = H·c²` for every row, so the ridge
/// is exactly `λ·H·c²` — the closed form the reduction has to reproduce.
#[test]
fn matches_closed_form() -> anyhow::Result<()> {
    let dev = cpu();
    for &(rows, h, c, lambda) in &[
        (7usize, 3usize, 2.0f32, 1.0f64),
        (128, 32, 0.1, 1.0),
        (18_000, 32, 0.05, 2.5),
    ] {
        let table = (Tensor::ones((rows, h), DType::F32, &dev)? * f64::from(c))?;
        let got = embedding_ridge(&table, lambda)?.to_scalar::<f32>()?;
        let want = (lambda * f64::from(h as f32) * f64::from(c * c)) as f32;
        assert!(
            (got - want).abs() <= 1e-4 * want.abs().max(1.0),
            "rows={rows} h={h}: got {got}, want {want}"
        );
    }
    Ok(())
}

/// Invariant to row count: stacking the same rows twice must not change the
/// penalty. This is the property the row-MEAN is there to provide, and the one
/// the old `mean_all` also had — so on its own it does not catch the bug.
#[test]
fn invariant_to_row_count() -> anyhow::Result<()> {
    let dev = cpu();
    let table = Tensor::randn(0f32, 1f32, (64, 16), &dev)?;
    let doubled = Tensor::cat(&[&table, &table], 0)?;
    let a = embedding_ridge(&table, 1.0)?.to_scalar::<f32>()?;
    let b = embedding_ridge(&doubled, 1.0)?.to_scalar::<f32>()?;
    assert!((a - b).abs() <= 1e-5 * a.abs().max(1.0), "{a} vs {b}");
    Ok(())
}

/// The load-bearing one, and the one that fails on the old reduction: at a
/// fixed per-element scale, DOUBLING the latent dim must double the penalty,
/// because each row's squared norm doubles. `mean_all` returns the same value
/// for both — that is precisely the `÷H` dilution.
#[test]
fn scales_with_latent_dim_not_diluted_by_it() -> anyhow::Result<()> {
    let dev = cpu();
    let narrow = (Tensor::ones((100, 16), DType::F32, &dev)? * 0.3)?;
    let wide = (Tensor::ones((100, 32), DType::F32, &dev)? * 0.3)?;
    let a = embedding_ridge(&narrow, 1.0)?.to_scalar::<f32>()?;
    let b = embedding_ridge(&wide, 1.0)?.to_scalar::<f32>()?;
    assert!(
        (b / a - 2.0).abs() < 1e-4,
        "doubling H must double the ridge, got {a} -> {b} (ratio {})",
        b / a
    );

    // Same statement as a direct contrast with the reduction this replaced,
    // so the regression is named in the failure output rather than inferred.
    let diluted_a = narrow.sqr()?.mean_all()?.to_scalar::<f32>()?;
    let diluted_b = wide.sqr()?.mean_all()?.to_scalar::<f32>()?;
    assert!(
        (diluted_a - diluted_b).abs() < 1e-6,
        "sanity: mean_all is H-invariant, which is the bug being pinned"
    );
    Ok(())
}

/// The gradient is what the optimizer sees, so pin it too: `∂/∂e = 2λ·e/rows`,
/// with NO `H` in the denominator.
#[test]
fn gradient_is_independent_of_latent_dim() -> anyhow::Result<()> {
    let dev = cpu();
    let rows = 50usize;
    for &h in &[8usize, 64] {
        let table = candle_util::candle_core::Var::from_tensor(
            &(Tensor::ones((rows, h), DType::F32, &dev)? * 0.5)?,
        )?;
        let ridge = embedding_ridge(table.as_tensor(), 1.0)?;
        let grads = ridge.backward()?;
        let g = grads
            .get(&table)
            .expect("ridge must produce a gradient on the table")
            .flatten_all()?
            .to_vec1::<f32>()?;
        let want = 2.0 * 0.5 / rows as f32; // 2λe/rows
        assert!(
            (g[0] - want).abs() < 1e-6,
            "h={h}: per-element grad {}, want {want}",
            g[0]
        );
    }
    Ok(())
}
