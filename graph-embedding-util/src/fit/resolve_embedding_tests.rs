use super::*;

/// Synthetic block recovery: K=2 topics, D=6 genes (0,1,2 mark topic 0;
/// 3,4,5 mark topic 1), N=40 cells (first 20 in topic 0). θ ≈ one-hot;
/// each cell emits counts only on its topic's marker genes. After
/// training, each topic's cells must score their OWN markers higher than
/// the other topic's markers in the shared space (Z_c · ρ_g).
#[test]
fn topic_blocks_recover_marker_cell_affinity() -> anyhow::Result<()> {
    let (k, d, n) = (2usize, 6usize, 40usize);
    let mut theta = Mat::zeros(n, k);
    for i in 0..n {
        let t = usize::from(i >= n / 2);
        theta[(i, t)] = 0.9;
        theta[(i, 1 - t)] = 0.1;
    }
    let mut edge_cell = Vec::new();
    let mut edge_gene = Vec::new();
    let mut edge_w = Vec::new();
    let mut gene_marginal = vec![0f64; d];
    for i in 0..n {
        let markers: [u32; 3] = if i < n / 2 { [0, 1, 2] } else { [3, 4, 5] };
        for &g in &markers {
            edge_cell.push(i as u32);
            edge_gene.push(g);
            edge_w.push(5.0);
            gene_marginal[g as usize] += 5.0;
        }
    }

    let inputs = RestTrainInputs {
        theta_aligned: theta,
        edge_gene,
        edge_cell,
        edge_w,
        gene_marginal,
        n_genes: d,
    };
    let dev = Device::Cpu;
    let trained = train_rest(
        &inputs,
        &RestConfig {
            embedding_dim: 4,
            epochs: 200,
            batches_per_epoch: 20,
            batch_size: 32,
            num_negatives: 4,
            learning_rate: 0.05,
            weight_decay: 0.0,
            neg_alpha: 0.75,
            seed: 42,
            dev: &dev,
            stop: Arc::new(AtomicBool::new(false)),
        },
    )?;

    let score = |c: usize, g: usize| -> f32 { trained.z.row(c).dot(&trained.rho.row(g)) };
    let mean_score = |cells: std::ops::Range<usize>, genes: [usize; 3]| -> f32 {
        let mut s = 0f32;
        let mut cnt = 0usize;
        for c in cells {
            for &g in &genes {
                s += score(c, g);
                cnt += 1;
            }
        }
        s / cnt as f32
    };

    let t0_own = mean_score(0..20, [0, 1, 2]);
    let t0_other = mean_score(0..20, [3, 4, 5]);
    let t1_own = mean_score(20..40, [3, 4, 5]);
    let t1_other = mean_score(20..40, [0, 1, 2]);

    assert!(
        t0_own > t0_other,
        "topic-0 cells should score own markers higher: {t0_own} vs {t0_other}"
    );
    assert!(
        t1_own > t1_other,
        "topic-1 cells should score own markers higher: {t1_own} vs {t1_other}"
    );
    assert!(
        trained.loss_trace.last().unwrap() < trained.loss_trace.first().unwrap(),
        "loss should decrease: {:?} → {:?}",
        trained.loss_trace.first(),
        trained.loss_trace.last()
    );
    Ok(())
}
