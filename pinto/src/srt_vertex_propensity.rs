use crate::srt_cell_pairs::*;
use crate::srt_common::*;

pub trait SrtVertPropOps {
    /// Take `edge_membership` (of some clustering algorithm) and
    /// estimate vertex propensity of the clustering results
    /// * `edge_membership` - `1 x num of edge` vector
    /// * `block_size` - how many edges per job
    /// * Returns `k x n` where `k`, clusters, `n`, vertices
    fn vertex_propensity(
        &self,
        edge_membership: &[usize],
        block_size: usize,
    ) -> anyhow::Result<Mat>;
}

impl SrtVertPropOps for SrtCellPairs<'_> {
    fn vertex_propensity(
        &self,
        edge_membership: &[usize],
        block_size: usize,
    ) -> anyhow::Result<Mat> {
        assert_eq!(self.num_pairs(), edge_membership.len());

        let nvertices = self.data.num_columns()?;

        let max_k = *edge_membership.iter().max().ok_or(anyhow::anyhow!(
            "unable to figure out the number of vertex groups"
        ))? + 1;

        let mut prop_kn = Mat::zeros(max_k, nvertices);

        self.visit_pairs_by_block(
            &count_incidence_visitor,
            &edge_membership,
            &mut prop_kn,
            block_size,
        )?;
        prop_kn.sum_to_one_columns_inplace();

        Ok(prop_kn)
    }
}

fn count_incidence_visitor(
    bound: (usize, usize),
    data: &SrtCellPairs,
    edge_membership: &[usize],
    arc_count_kn: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let pairs = &data.pairs[lb..ub];
    let membership = &edge_membership[lb..ub];

    let mut count_kn = arc_count_kn.lock().expect("lock count nk");

    for (pp, &k) in pairs.iter().zip(membership) {
        if pp.left < count_kn.ncols() {
            count_kn.column_mut(pp.left)[k] += 1.0;
        }
        if pp.right < count_kn.ncols() {
            count_kn.column_mut(pp.right)[k] += 1.0;
        }
    }
    Ok(())
}
