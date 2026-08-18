use super::*;

/// Build a level with known counts and a hand-written super-edge list. Column
/// `p` of `counts` is filled with `p + 1`, so any mis-summing shows up as a
/// wrong integer rather than a plausible-looking float.
fn level(super_edges: Vec<(usize, usize)>, n_pb: usize, n_genes: usize) -> LevelPseudobulk {
    let counts = Mat::from_fn(n_genes, n_pb, |_, p| (p + 1) as f32);
    LevelPseudobulk {
        cell_labels: Vec::new(),
        counts,
        // One fine edge per super-edge keeps `fine_to_super` well-formed; the
        // collapse does not read it, but the invariant should hold anyway.
        fine_to_super: (0..super_edges.len()).map(Some).collect(),
        super_edges,
        e_pb: Mat::zeros(n_pb, 1),
        pb_offset: 0,
    }
}

/// Intra-group fine edges are reported as `None`, not as a super-edge.
#[test]
fn internal_fine_edges_have_no_super_edge() {
    let mut lvl = level(vec![(0, 1)], 2, 3);
    lvl.fine_to_super = vec![Some(0), None, None];
    assert_eq!(lvl.n_internal_fine_edges(), 2);
    assert_eq!(lvl.n_super_edges(), 1);
}

/////////////////////////////////////
// the two-block splice alternation //
/////////////////////////////////////

const H: usize = 3;
const N_PB: usize = 12;

/// Three genes with deliberately different track coverage:
/// `g0` both tracks, `g1` spliced only, `g2` unspliced only. Only `g0` can pin
/// the nascent-minus-mature contrast.
fn splice_state() -> SelectionState {
    let counts = |seed: usize| -> Vec<(u32, f32)> {
        (0..N_PB as u32)
            .map(|p| (p, 1.0 + ((p as usize * 7 + seed) % 5) as f32))
            .collect()
    };
    let tracks = TrackPos {
        spliced: vec![counts(0), counts(1), Vec::new()],
        unspliced: vec![counts(2), Vec::new(), counts(3)],
    };
    let delta_identified = tracks
        .spliced
        .iter()
        .zip(&tracks.unspliced)
        .map(|(s, u)| !s.is_empty() && !u.is_empty())
        .collect();
    SelectionState {
        tracks,
        delta_identified,
        strength: DeltaStrength {
            median_counts: 36.0,
            median_pb_detected: N_PB as f32,
            total_pb: N_PB,
        },
        partition: (0..N_PB as u32).collect(),
        b_flat: vec![0.0; N_PB],
        level_maps: Vec::new(),
        total_pb: N_PB,
        n_genes: 3,
        dim: H,
    }
}

/// A frozen pseudobulk side with structure on every dim, so the blocks have
/// something to select on.
fn frozen_side_values() -> Vec<f32> {
    (0..N_PB * H)
        .map(|i| {
            let (p, k) = (i / H, i % H);
            if p % H == k {
                1.0
            } else {
                -0.25
            }
        })
        .collect()
}

fn run(nested: bool) -> (Selection, ChainWarm) {
    let state = splice_state();
    let e = frozen_side_values();
    let side = FrozenSide {
        e: &e,
        b: &state.b_flat,
        h: H,
    };
    state.sample_two_block(
        &side,
        &SelectArgs {
            sweeps: 12,
            burnin: 4,
            seed: 42,
            nested_delta: nested,
        },
        ChainWarm::default(),
    )
}

/// `δ` is identified only by the contrast, so a one-track gene must come back
/// NaN — not a number drawn from the prior.
///
/// Break it by dropping the NaN mask and this fails on `g1`: the sampler returns
/// a perfectly plausible finite PIP for a gene whose delta entered no likelihood
/// term at all.
#[test]
fn a_one_track_gene_gets_nan_not_a_prior_draw() {
    let (sel, _) = run(true);
    let d = sel.delta.expect("channelized input reports delta");

    assert_eq!(d.identified, vec![true, false, false]);
    assert_eq!(d.n_identified(), 1);

    for k in 0..H {
        assert!(d.pip[k].is_finite(), "g0 dim {k} is measured");
        assert!(d.mean[k].is_finite());
        for g in 1..3 {
            assert!(d.pip[g * H + k].is_nan(), "g{g} dim {k} pip must be NaN");
            assert!(d.mean[g * H + k].is_nan(), "g{g} dim {k} mean must be NaN");
        }
    }
    // beta is unaffected: every gene has counts on at least one track, so the
    // identity loading stays a measurement for all three.
    assert!(sel.pip.iter().all(|v| v.is_finite()));
    assert!(sel.mean_beta.iter().all(|v| v.is_finite()));
}

/// The nested gate is a hard veto, not a prior nudge: `z_δ = 1` where `z_β = 0`
/// must be unreachable.
///
/// Break it by dropping the `with_z_allowed` call and this fails — the two
/// independent spike-and-slabs happily put mass on the forbidden corner.
#[test]
fn the_nested_gate_never_admits_delta_where_beta_is_off() {
    let (_, warm) = run(true);
    let zb = warm.z_beta.expect("beta state carried out");
    let zd = warm.z_delta.expect("delta state carried out");
    assert_eq!(zb.len(), 3 * H);
    assert_eq!(zd.len(), zb.len());
    for (i, (&b, &d)) in zb.iter().zip(&zd).enumerate() {
        assert!(b || !d, "coordinate {i}: z_delta on where z_beta is off");
    }
}

/// The escape hatch has to actually reach the sampler, and it must not change
/// what is identified — only which corners the chain may visit.
#[test]
fn the_independent_gate_runs_and_keeps_the_same_identifiability() {
    let (sel, warm) = run(false);
    let d = sel.delta.as_ref().expect("delta");
    assert_eq!(d.identified, vec![true, false, false]);
    assert!(warm.z_delta.is_some() && warm.e_delta.is_some());
    // The hyper-chains are per OUTER sweep, so they carry the retained draws
    // rather than one value — without that, mixing is unreportable.
    assert_eq!(d.sigma_diag.len(), H);
    assert!(
        sel.n_kept > 0,
        "retained sweeps are what the diagnostic is over"
    );
}

/// A chain that resumes must not restart. Feeding the previous run's state back
/// in has to be accepted and carried through.
#[test]
fn a_warm_start_is_carried_rather_than_discarded() {
    let state = splice_state();
    let e = frozen_side_values();
    let side = FrozenSide {
        e: &e,
        b: &state.b_flat,
        h: H,
    };
    let args = SelectArgs {
        sweeps: 6,
        burnin: 2,
        seed: 7,
        nested_delta: true,
    };
    let (_, warm1) = state.sample_two_block(&side, &args, ChainWarm::default());
    let e_delta_in = warm1.e_delta.clone().expect("delta loadings");
    let (_, warm2) = state.sample_two_block(&side, &args, warm1);

    assert_eq!(warm2.e_delta.as_ref().unwrap().len(), e_delta_in.len());
    assert!(warm2.z_beta.is_some() && warm2.z_delta.is_some());
}
