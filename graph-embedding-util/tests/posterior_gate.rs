//! Stage-A validation for the per-gene SuSiE gate posterior
//! (`graph_embedding_util::posterior::gate`).
//!
//! Plants genes that load a SINGLE known embedding dim (one-hot loadings) plus a
//! block of null genes with no loading, generates Poisson counts against a fixed
//! cell side, and checks the exact gate posterior:
//!   1. a signal gene's max-PIP dim is the dim it was planted on;
//!   2. a null gene's loading posterior is near zero (small ‖mean_beta‖).
//!
//! The cell side is frozen at truth, so there is no gauge freedom and the
//! per-dim PIP is directly meaningful.

use graph_embedding_util::posterior::{gate_posterior, FrozenSide, GateConfig, NodeTerm};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson, StandardNormal};

const H: usize = 5;
const N_CELL: usize = 80;
const N_SIGNAL: usize = 15; // signal genes: 3 per dim
const N_NULL: usize = 10;

fn randn(sd: f32, rng: &mut SmallRng) -> f32 {
    let z: f64 = StandardNormal.sample(rng);
    z as f32 * sd
}

#[test]
fn gate_posterior_recovers_loaded_dim_and_nulls() {
    let mut rng = SmallRng::seed_from_u64(424242);

    // Cell side: random embeddings, negative bias floor (modest rates).
    let e_cell: Vec<f32> = (0..N_CELL * H).map(|_| randn(0.7, &mut rng)).collect();
    let b_cell: Vec<f32> = (0..N_CELL).map(|_| -0.5).collect();

    // Genes. Signal gene g loads dim (g % H) with a strong one-hot loading;
    // null genes have a zero loading (pure bias). Record the planted dim.
    let n_genes = N_SIGNAL + N_NULL;
    let mut planted_dim = vec![usize::MAX; n_genes]; // MAX = null
    let mut e_gene = vec![0.0f32; n_genes * H];
    let b_gene = vec![-0.3f32; n_genes];
    for g in 0..N_SIGNAL {
        let d = g % H;
        planted_dim[g] = d;
        e_gene[g * H + d] = 1.6; // strong single-dim loading
    }
    // null genes (N_SIGNAL..n_genes) keep the zero loading.

    // Counts n_gc ~ Poisson(exp(⟨e_g, e_c⟩ + b_g + b_c)); keep nonzero edges.
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
            let s = dot + f64::from(b_gene[g]) + f64::from(b_cell[c]);
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

    let cfg = GateConfig::new(2000, 800, 11);
    let post = gate_posterior(&nodes, &side, &cfg);
    assert_eq!(post.len(), n_genes);

    // (1) Signal genes: argmax PIP == planted dim.
    let mut correct = 0usize;
    for g in 0..N_SIGNAL {
        let argmax = post[g]
            .pip
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        if argmax == planted_dim[g] {
            correct += 1;
        }
    }
    // Allow one miss to a co-linear dim; the rest must be exact.
    assert!(
        correct >= N_SIGNAL - 1,
        "signal genes should load their planted dim ({correct}/{N_SIGNAL} correct)"
    );

    // (2) Null genes: near-zero loading vs signal genes' loading.
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let signal_norm: f32 =
        (0..N_SIGNAL).map(|g| norm(&post[g].mean_beta)).sum::<f32>() / N_SIGNAL as f32;
    let null_norm: f32 = (N_SIGNAL..n_genes)
        .map(|g| norm(&post[g].mean_beta))
        .sum::<f32>()
        / N_NULL as f32;
    assert!(
        null_norm < 0.5 * signal_norm,
        "null genes should have a much smaller loading (null={null_norm:.3}, signal={signal_norm:.3})"
    );
}
