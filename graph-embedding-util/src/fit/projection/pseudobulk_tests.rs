//! Unit tests for the pseudobulk phase-2 velocity readout.

use super::project_pbs_phase2;
use crate::data::UnifiedData;
use candle_util::candle_core::Device;

/// Cosine alignment of two equal-length vectors.
fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

// Two pb nodes with planted identity θ* and velocity δ*. Spliced rows carry
// counts at the noiseless rate exp(⟨e_f, θ*⟩ + b_f + b_c); unspliced rows at
// exp(⟨e_f, θ*+δ*⟩ + b_f + b_c). The dual pb projection must recover θ* from the
// spliced aggregate and δ* from the unspliced aggregate (θ* held fixed).
#[test]
fn pb_projection_recovers_theta_and_delta() {
    let h = 4;
    let n_spliced = 8;
    let n_feat = n_spliced * 2; // rows 0..8 spliced, 8..16 unspliced
    let unspliced_rows: Vec<bool> = (0..n_feat).map(|f| f >= n_spliced).collect();

    // Deterministic frozen dictionary e_f / b_f.
    let mut e = vec![0f32; n_feat * h];
    let mut b = vec![0f32; n_feat];
    for f in 0..n_feat {
        for k in 0..h {
            e[f * h + k] = (((f * 7 + k * 13) % 11) as f32 / 11.0) - 0.5;
        }
        b[f] = (((f * 5) % 7) as f32 / 7.0) - 0.3;
    }

    // Per-pb planted identity + velocity.
    let theta_star = [[0.6f32, -0.4, 0.3, 0.2], [-0.3, 0.5, -0.2, 0.4]];
    let delta_star = [[0.25f32, -0.3, 0.15, 0.2], [-0.2, 0.1, 0.3, -0.25]];
    let b_c = [0.2f32, -0.1];
    let n_pb = theta_star.len();

    let rate = |f: usize, latent: &[f32; 4], bc: f32| -> f32 {
        let ef = &e[f * h..(f + 1) * h];
        let s: f32 = ef.iter().zip(latent).map(|(a, c)| a * c).sum::<f32>() + b[f] + bc;
        s.exp()
    };
    let counts = nalgebra::DMatrix::<f32>::from_fn(n_feat, n_pb, |f, p| {
        if unspliced_rows[f] {
            // nascent state = θ* + δ*
            let nascent = [
                theta_star[p][0] + delta_star[p][0],
                theta_star[p][1] + delta_star[p][1],
                theta_star[p][2] + delta_star[p][2],
                theta_star[p][3] + delta_star[p][3],
            ];
            rate(f, &nascent, b_c[p])
        } else {
            rate(f, &theta_star[p], b_c[p])
        }
    });

    let names: Vec<Box<str>> = (0..n_feat)
        .map(|f| format!("g{f}").into_boxed_str())
        .collect();
    let pb = UnifiedData::from_pseudobulks(&counts, names, (0..n_feat).collect()).unwrap();

    let levels = project_pbs_phase2(&e, &b, h, &[pb], &unspliced_rows, 1e-3, &Device::Cpu).unwrap();
    assert_eq!(levels.len(), 1);
    let lv = &levels[0];
    assert_eq!(lv.n_pb, n_pb);
    assert_eq!(lv.theta.len(), n_pb * h);
    assert_eq!(lv.delta.len(), n_pb * h);

    for p in 0..n_pb {
        let th = &lv.theta[p * h..(p + 1) * h];
        let dl = &lv.delta[p * h..(p + 1) * h];
        let ct = cos(th, &theta_star[p]);
        let cd = cos(dl, &delta_star[p]);
        assert!(ct > 0.97, "pb {p} identity misaligned (cos={ct:.3})");
        assert!(cd > 0.97, "pb {p} velocity misaligned (cos={cd:.3})");
    }
}

// A pb node with no unspliced edges (spliced-only) yields a defined θ but a
// zero δ row (velocity undefined), and the buffer stays the right shape.
#[test]
fn pb_projection_zero_delta_without_unspliced() {
    let h = 3;
    let n_feat = 6;
    // Every row spliced ⇒ no unspliced aggregate for any pb.
    let unspliced_rows = vec![false; n_feat];
    let mut e = vec![0f32; n_feat * h];
    let mut b = vec![0f32; n_feat];
    for f in 0..n_feat {
        for k in 0..h {
            e[f * h + k] = (((f * 5 + k * 3) % 7) as f32 / 7.0) - 0.5;
        }
        b[f] = 0.0;
    }
    let counts = nalgebra::DMatrix::<f32>::from_fn(n_feat, 1, |f, _| f as f32 + 1.0);
    let names: Vec<Box<str>> = (0..n_feat)
        .map(|f| format!("g{f}").into_boxed_str())
        .collect();
    let pb = UnifiedData::from_pseudobulks(&counts, names, (0..n_feat).collect()).unwrap();

    let levels = project_pbs_phase2(&e, &b, h, &[pb], &unspliced_rows, 1e-2, &Device::Cpu).unwrap();
    let lv = &levels[0];
    assert_eq!(lv.delta, vec![0.0; h]);
    // Identity is non-trivial (there were spliced edges).
    assert!(lv.theta.iter().any(|x| x.abs() > 1e-6));
}
