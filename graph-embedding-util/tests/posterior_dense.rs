//! Does the single-effect gate posterior "collapse" when the truth is DENSE?
//!
//! On real data the gate posterior concentrates ~40% of genes onto two dims while
//! the fitted dictionary spreads evenly, and PIP-argmax agrees with
//! dictionary-argmax at chance. Two competing explanations:
//!
//!   (A) the sampler is broken / mis-specified, or
//!   (B) the comparison is ill-posed — a SuSiE single-effect answers "if this gene
//!       loaded exactly one dim, which?", and a dense loading has no such answer,
//!       so its argmax is arbitrary and agreement *should* sit at chance.
//!
//! This test discriminates them on planted truth, where the answer is known. Two
//! gene blocks share one generator and one frozen cell side:
//!
//!   * SPARSE genes load a single dim (the model's own assumption);
//!   * DENSE genes load every dim with comparable magnitude, like the real fit.
//!
//! MEASURED (2026-07-27): sparse recovery 1.000; dense recovery **0.650 against a
//! 0.125 chance rate** — degraded, but 5× chance; dense top-2 dim share 0.45 (vs
//! 0.25 uniform), peaking on the dim given 2× spread. So (B) is FALSE in its
//! strong form: a single effect does find the dominant coordinate of a dense
//! loading, and the frozen geometry biases *which* dim without destroying
//! recovery.
//!
//! That matters for reading real data: on 12k BMMC the same code scores 0.079
//! against 0.062 chance — no recovery at all, which this simulation does NOT
//! reproduce. Whatever breaks there is absent here, so it is not the
//! single-effect-vs-dense mismatch. This test pins that conclusion down: if a
//! future change makes dense recovery collapse to chance, the difference is in
//! the sampler, not the estimand.

use graph_embedding_util::posterior::{gate_posterior, FrozenSide, GateConfig, NodeTerm};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson, StandardNormal};

const H: usize = 8;
const N_CELL: usize = 300;
const N_SPARSE: usize = 40;
const N_DENSE: usize = 40;

fn randn(sd: f32, rng: &mut SmallRng) -> f32 {
    let z: f64 = StandardNormal.sample(rng);
    z as f32 * sd
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

#[test]
fn dense_truth_is_degraded_but_still_recovered() {
    let mut rng = SmallRng::seed_from_u64(20260727);

    // Frozen cell side. Deliberately ANISOTROPIC: dim 0 carries twice the spread
    // of the rest, which is what a real embedding looks like and what a
    // geometry-driven collapse would key on.
    let mut e_cell = vec![0.0f32; N_CELL * H];
    for c in 0..N_CELL {
        for k in 0..H {
            let sd = if k == 0 { 1.4 } else { 0.7 };
            e_cell[c * H + k] = randn(sd, &mut rng);
        }
    }
    let b_cell: Vec<f32> = (0..N_CELL).map(|_| randn(0.4, &mut rng) - 0.5).collect();

    // Genes: a sparse block (one-hot) then a dense block (all dims loaded).
    let n_genes = N_SPARSE + N_DENSE;
    let mut e_gene = vec![0.0f32; n_genes * H];
    let mut planted = vec![0usize; n_genes];
    for g in 0..N_SPARSE {
        let d = g % H;
        planted[g] = d;
        e_gene[g * H + d] = 1.6;
    }
    for g in N_SPARSE..n_genes {
        // Dense: every dim loaded at comparable magnitude. The `planted` dim is
        // merely the largest coordinate — for a dense vector that is a weak,
        // noise-dominated label, which is exactly the point.
        for k in 0..H {
            e_gene[g * H + k] = randn(0.55, &mut rng);
        }
        planted[g] = argmax(
            &e_gene[g * H..(g + 1) * H]
                .iter()
                .map(|x| x.abs())
                .collect::<Vec<_>>(),
        );
    }

    // n_gc ~ Poisson(exp(⟨e_g,e_c⟩ + b_g + b_c)) — the same generator both blocks
    // share, so any difference in recovery is about the planted STRUCTURE.
    let b_gene = vec![-0.3f32; n_genes];
    let mut pos_by_gene: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_genes];
    for g in 0..n_genes {
        let eg = &e_gene[g * H..(g + 1) * H];
        for c in 0..N_CELL {
            let ec = &e_cell[c * H..(c + 1) * H];
            let dot: f64 = eg
                .iter()
                .zip(ec)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let s = (dot + f64::from(b_gene[g]) + f64::from(b_cell[c])).min(8.0);
            let n = Poisson::new(s.exp()).unwrap().sample(&mut rng);
            if n > 0.0 {
                pos_by_gene[g].push((c as u32, n as f32));
            }
        }
    }

    let side = FrozenSide {
        e: &e_cell,
        b: &b_cell,
        h: H,
    };
    let all_cells: Vec<u32> = (0..N_CELL as u32).collect();
    let nodes: Vec<NodeTerm> = pos_by_gene
        .iter()
        .map(|pos| NodeTerm::new(pos, &all_cells, 1.0))
        .collect();
    let post = gate_posterior(&nodes, &side, &GateConfig::new(1500, 600, 5));

    let hit = |range: std::ops::Range<usize>| -> f64 {
        let n = range.len();
        range
            .filter(|&g| argmax(&post[g].pip) == planted[g])
            .count() as f64
            / n as f64
    };
    let sparse_acc = hit(0..N_SPARSE);
    let dense_acc = hit(N_SPARSE..n_genes);

    // Concentration of the DENSE block's argmax: if the collapse is driven by the
    // frozen geometry rather than the planted loadings, dense genes pile onto a
    // couple of shared dims even though their truths are independent draws.
    let mut usage = vec![0usize; H];
    for g in N_SPARSE..n_genes {
        usage[argmax(&post[g].pip)] += 1;
    }
    let mut sorted = usage.clone();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let top2 = (sorted[0] + sorted[1]) as f64 / N_DENSE as f64;

    eprintln!(
        "sparse recovery {sparse_acc:.3} | dense recovery {dense_acc:.3} \
         (chance {:.3}) | dense top-2 dim share {top2:.3} | dense usage {usage:?}",
        1.0 / H as f64
    );

    // The discriminator. Sparse recovery must be high — the sampler works, and
    // both blocks share the code path, so this rules out (A).
    assert!(
        sparse_acc >= 0.75,
        "single-dim truth must be recovered (got {sparse_acc:.3}); \
         if this fails the sampler itself is suspect, not the estimand"
    );
    // Dense recovery is DEGRADED but stays well clear of chance. Both bounds
    // matter: the upper one says a dense loading is genuinely harder for a single
    // effect; the lower one says it is still recovered, which is what rules the
    // estimand out as the explanation for the real-data collapse.
    assert!(
        dense_acc < sparse_acc,
        "dense truth should be harder than one-hot truth \
         (sparse {sparse_acc:.3} vs dense {dense_acc:.3})"
    );
    assert!(
        dense_acc > 3.0 / H as f64,
        "dense truth must still be recovered well above chance (got {dense_acc:.3}, \
         chance {:.3}) — at chance, the real-data disagreement would be explained by \
         the estimand rather than by the data",
        1.0 / H as f64
    );
}
