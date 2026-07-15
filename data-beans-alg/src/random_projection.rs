#![allow(dead_code)]

use data_beans::sparse_data_visitors::*;
use data_beans::sparse_io_stack::SparseIoStack;
use data_beans::sparse_io_vector::SparseIoVec;
use matrix_util::dmatrix_util::*;
use matrix_util::rand_util::mix_seed;
use std::sync::{Arc, Mutex};

use log::{info, warn};
use matrix_util::traits::*;
use matrix_util::utils::*;
use nalgebra::DVector;

/// Bundle passed as `SharedIn` so the visitor signature stays compatible
/// with `visit_columns_by_block`. The basis is stored as `K × D`
/// (transposed) so the inner kernel accesses contiguous K-vectors per
/// non-zero, instead of stride-D reads across `D × K` column-major
/// storage.
struct ProjVisitorIn<'a> {
    basis_kd: &'a nalgebra::DMatrix<f32>,
}

pub struct RandColProjOut {
    pub basis: nalgebra::DMatrix<f32>,
    pub proj: nalgebra::DMatrix<f32>,
}

pub struct RandRowProjOut {
    pub basis: nalgebra::DMatrix<f32>,
    pub proj: nalgebra::DMatrix<f32>,
}

/// Default random-basis seed for the non-seeded projection entry points.
///
/// The random-projection basis only needs to be *stable* (a fixed
/// Johnson–Lindenstrauss draw is still a valid projection), so callers that do
/// not thread their own `--seed` get a pinned, reproducible basis for free.
/// Callers that want `--seed` to also drive the projection use the `_seeded`
/// variants (e.g. the shared `graph_embedding_util::fit` gem/bge path).
pub const DEFAULT_PROJECTION_SEED: u64 = 0x50524F4A_50524F4A; // "PROJPROJ"

pub trait RandProjOps {
    ///
    /// Create K x ncol projection by concatenating data across
    /// columns and returns the random basis matrix `D` x `target_dim`
    /// and the projection result `target_dim` x `N`
    ///
    /// # Arguments
    /// * `target_dim`: target dimensionality
    /// * `block_size`: block size for parallel computation
    ///
    /// # Output
    /// * `basis`: `D x K` random basis matrix
    /// * `proj`: `K x N` projection results
    ///
    /// Uses [`DEFAULT_PROJECTION_SEED`] for the basis; for a caller-controlled
    /// seed see [`RandProjOps::project_columns_with_batch_correction_seeded`].
    fn project_columns(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
    ) -> anyhow::Result<RandColProjOut> {
        self.project_columns_with_batch_correction::<usize>(target_dim, block_size, None)
    }

    ///
    /// Create K x ncol projection by concatenating data across
    /// columns and returns the random basis matrix `D` x `target_dim`
    /// and the projection result `target_dim` x `N`
    ///
    /// # Arguments
    /// * `target_dim`: target dimensionality
    /// * `block_size`: block size for parallel computation
    ///
    /// # Output
    /// * `basis`: `D x K` random basis matrix
    /// * `proj`: `K x N` projection results
    ///
    /// Uses [`DEFAULT_PROJECTION_SEED`]; delegates to
    /// [`RandProjOps::project_columns_with_batch_correction_seeded`].
    fn project_columns_with_batch_correction<T>(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
        batch_membership: Option<&[T]>,
    ) -> anyhow::Result<RandColProjOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        self.project_columns_with_batch_correction_seeded(
            target_dim,
            block_size,
            batch_membership,
            DEFAULT_PROJECTION_SEED,
        )
    }

    /// Seeded variant of [`RandProjOps::project_columns_with_batch_correction`]:
    /// `seed` pins the random basis so identical-config runs are byte-identical.
    fn project_columns_with_batch_correction_seeded<T>(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
        batch_membership: Option<&[T]>,
        seed: u64,
    ) -> anyhow::Result<RandColProjOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;

    /// Like `project_columns_with_batch_correction` but with per-row
    /// (per-feature) weights applied to the random basis. Features with
    /// weight 0 are excluded from the projection geometry.
    ///
    /// Uses [`DEFAULT_PROJECTION_SEED`]; delegates to
    /// [`RandProjOps::project_columns_weighted_seeded`].
    fn project_columns_weighted<T>(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
        batch_membership: Option<&[T]>,
        row_weights: &[f32],
    ) -> anyhow::Result<RandColProjOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        self.project_columns_weighted_seeded(
            target_dim,
            block_size,
            batch_membership,
            row_weights,
            DEFAULT_PROJECTION_SEED,
        )
    }

    /// Seeded variant of [`RandProjOps::project_columns_weighted`]: `seed` pins
    /// the random basis so identical-config runs are byte-identical.
    fn project_columns_weighted_seeded<T>(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
        batch_membership: Option<&[T]>,
        row_weights: &[f32],
        seed: u64,
    ) -> anyhow::Result<RandColProjOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString;

    /// Assign each column/cell to random group by binary encoding
    ///
    /// # Arguments
    ///
    /// * `proj_kn` - random projection matrix (feature x column/cell)
    ///
    /// * `num_features` - number of top features (if None, use all of them)
    ///
    fn partition_columns_to_groups(
        &mut self,
        proj_kn: &nalgebra::DMatrix<f32>,
        num_features: Option<usize>,
        ncols_per_group: Option<usize>,
    ) -> anyhow::Result<usize>;

    // Take samples assigned
    // fn groups_assigned(&self) -> anyhow::Result<&Vec<usize>>;
}

/// column-wise visitor for random projection
fn project_columns_visitor(
    job: (usize, usize),
    data_vec: &SparseIoVec,
    shared_in: &ProjVisitorIn<'_>,
    arc_proj_kn: Arc<Mutex<&mut nalgebra::DMatrix<f32>>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let basis_kd = shared_in.basis_kd;
    let target_dim = basis_kd.nrows();
    let n_block = ub - lb;

    let mut xx_dm = data_vec.read_columns_csc(lb..ub)?;
    for x in xx_dm.values_mut() {
        *x = x.ln_1p();
    }
    xx_dm.normalize_columns_inplace();

    // Direct K×N kernel: chunk[:, j] = Σᵢ X[i,j] · B_kd[:, i]. One pass
    // over CSC non-zeros. Both column slices are unit-stride (contiguous),
    // so the SAXPY vectorizes; no sparse transpose, no dense transpose.
    let mut chunk = nalgebra::DMatrix::<f32>::zeros(target_dim, n_block);
    for (j, col) in xx_dm.col_iter().enumerate() {
        for (&i, &v) in col.row_indices().iter().zip(col.values()) {
            chunk.column_mut(j).axpy(v, &basis_kd.column(i), 1.0);
        }
    }

    let mut proj_kn = arc_proj_kn.lock().expect("proj_kn lock");
    proj_kn.columns_range_mut(lb..ub).copy_from(&chunk);
    Ok(())
}

impl RandProjOps for SparseIoStack {
    fn project_columns_with_batch_correction_seeded<T>(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
        batch_membership: Option<&[T]>,
        seed: u64,
    ) -> anyhow::Result<RandColProjOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let ncols = self.num_columns()?;
        let batch_membership = batch_membership.and_then(|x| x.get(0..ncols));

        let proj_vec = self
            .stack
            .iter()
            .enumerate()
            .map(|(m, data_vec)| -> anyhow::Result<_> {
                // Distinct sub-seed per stacked modality so same-sized modalities
                // do not share a basis.
                data_vec.project_columns_with_batch_correction_seeded(
                    target_dim,
                    block_size,
                    batch_membership,
                    mix_seed(seed, m as u64),
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let basis = concatenate_vertical(
            &proj_vec
                .iter()
                .map(|x| {
                    // info!("{} x {}", x.basis.nrows(), x.basis.ncols());
                    x.basis.to_owned()
                })
                .collect::<Vec<_>>(),
        )?;

        let proj = concatenate_vertical(
            &proj_vec
                .iter()
                .map(|x| {
                    // info!("{} x {}", x.proj.nrows(), x.proj.ncols());
                    x.proj.to_owned()
                })
                .collect::<Vec<_>>(),
        )?;

        Ok(RandColProjOut { basis, proj })
    }

    fn project_columns_weighted_seeded<T>(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
        batch_membership: Option<&[T]>,
        row_weights: &[f32],
        seed: u64,
    ) -> anyhow::Result<RandColProjOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let ncols = self.num_columns()?;
        let batch_membership = batch_membership.and_then(|x| x.get(0..ncols));

        // Interpret `row_weights` as the concatenated feature axis across
        // all stacked modalities, in `self.stack` order. Slice per
        // modality so each per-vec call sees weights of length equal to
        // its own `num_rows()`.
        let total_rows: usize = self.stack.iter().map(|v| v.num_rows()).sum();
        if row_weights.len() != total_rows {
            return Err(anyhow::anyhow!(
                "row_weights length {} does not match total feature count {} \
                 (sum of num_rows() across the stack)",
                row_weights.len(),
                total_rows
            ));
        }

        let mut offset = 0usize;
        let mut proj_vec = Vec::with_capacity(self.stack.len());
        for (m, data_vec) in self.stack.iter().enumerate() {
            let nrows = data_vec.num_rows();
            let slice = &row_weights[offset..offset + nrows];
            offset += nrows;
            proj_vec.push(data_vec.project_columns_weighted_seeded(
                target_dim,
                block_size,
                batch_membership,
                slice,
                mix_seed(seed, m as u64),
            )?);
        }

        let basis = concatenate_vertical(
            &proj_vec
                .iter()
                .map(|x| x.basis.to_owned())
                .collect::<Vec<_>>(),
        )?;
        let proj = concatenate_vertical(
            &proj_vec
                .iter()
                .map(|x| x.proj.to_owned())
                .collect::<Vec<_>>(),
        )?;

        Ok(RandColProjOut { basis, proj })
    }

    fn partition_columns_to_groups(
        &mut self,
        proj_kn: &nalgebra::DMatrix<f32>,
        num_sorting_features: Option<usize>,
        ncols_per_group: Option<usize>,
    ) -> anyhow::Result<usize> {
        let nn = proj_kn.ncols();
        if nn != self.num_columns()? {
            return Err(anyhow::anyhow!("number of columns mismatch"));
        }

        let target_kk = num_sorting_features.unwrap_or(proj_kn.nrows());
        let kk = proj_kn.nrows().min(target_kk).min(nn);

        let binary_codes = binary_sort_columns(proj_kn, kk)?;
        let max_group = *binary_codes
            .iter()
            .max()
            .ok_or(anyhow::anyhow!("unable to determine max element"))?;

        for x in self.stack.iter_mut() {
            x.assign_groups(&binary_codes, ncols_per_group);
        }

        Ok(max_group + 1)
    }
}

impl RandProjOps for SparseIoVec {
    fn project_columns_with_batch_correction_seeded<T>(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
        batch_membership: Option<&[T]>,
        seed: u64,
    ) -> anyhow::Result<RandColProjOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let nrows = self.num_rows();
        let ncols = self.num_columns();

        let mut proj_kn = nalgebra::DMatrix::<f32>::zeros(target_dim, ncols);

        let basis_dk = nalgebra::DMatrix::<f32>::rnorm_seeded(nrows, target_dim, seed);
        // Carry a K×D copy for the inner kernel so per-non-zero column
        // slices into the basis are contiguous (unit-stride SAXPY).
        let basis_kd = basis_dk.transpose();

        let shared_in = ProjVisitorIn {
            basis_kd: &basis_kd,
        };
        info!(
            "starting random projection loop (project_columns_with_batch_correction): \
             D={nrows}, N={ncols}, K={target_dim}, block_size={:?}",
            block_size
        );
        self.visit_columns_by_block(
            &project_columns_visitor,
            &shared_in,
            &mut proj_kn,
            block_size,
        )?;
        info!("finished random projection loop");

        if let Some(col_to_batch) = batch_membership {
            if col_to_batch.len() == ncols {
                let batches = partition_by_membership(col_to_batch, None);
                info!("adjusting batch biases ({} batches) ...", batches.len());
                for (_, cols) in batches.iter() {
                    let xx = subset_columns(&proj_kn, cols.iter().cloned())?
                        .transpose() // n x k
                        .centre_columns() // adjust the mean
                        .transpose(); // k x n
                    assign_columns(&xx, cols.iter().cloned(), &mut proj_kn);
                }
            } else {
                warn!(
                    "The batch membership size {} mismatches with the number of columns {} ...",
                    col_to_batch.len(),
                    ncols
                );
            }
        }

        let (lb, ub) = (-4., 4.);
        proj_kn.scale_columns_inplace();

        if proj_kn.max() > ub || proj_kn.min() < lb {
            // info!("Clamping values [{}, {}] after standardization", lb, ub);
            proj_kn.iter_mut().for_each(|x| {
                *x = x.clamp(lb, ub);
            });
            proj_kn.scale_columns_inplace();
        }
        Ok(RandColProjOut {
            basis: basis_dk,
            proj: proj_kn,
        })
    }

    /// Like `project_columns_with_batch_correction` but with per-row
    /// (per-feature) weights applied to the random basis. Features with
    /// weight 0 are excluded from the projection geometry.
    fn project_columns_weighted_seeded<T>(
        &self,
        target_dim: usize,
        block_size: Option<usize>,
        batch_membership: Option<&[T]>,
        row_weights: &[f32],
        seed: u64,
    ) -> anyhow::Result<RandColProjOut>
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let nrows = self.num_rows();
        let ncols = self.num_columns();

        assert_eq!(row_weights.len(), nrows, "row_weights length mismatch");

        let mut proj_kn = nalgebra::DMatrix::<f32>::zeros(target_dim, ncols);

        let mut basis_dk = nalgebra::DMatrix::<f32>::rnorm_seeded(nrows, target_dim, seed);

        // Zero out basis rows for filtered features
        for (r, &w) in row_weights.iter().enumerate() {
            if w <= 0.0 {
                basis_dk.row_mut(r).fill(0.0);
            } else if (w - 1.0).abs() > 1e-6 {
                basis_dk.row_mut(r).scale_mut(w);
            }
        }

        let basis_kd = basis_dk.transpose();
        let shared_in = ProjVisitorIn {
            basis_kd: &basis_kd,
        };
        info!(
            "starting random projection loop (project_columns_weighted): \
             D={nrows}, N={ncols}, K={target_dim}, block_size={:?}",
            block_size
        );
        self.visit_columns_by_block(
            &project_columns_visitor,
            &shared_in,
            &mut proj_kn,
            block_size,
        )?;
        info!("finished random projection loop");

        if let Some(col_to_batch) = batch_membership {
            if col_to_batch.len() == ncols {
                let batches = partition_by_membership(col_to_batch, None);
                info!("adjusting batch biases ({} batches) ...", batches.len());
                for (_, cols) in batches.iter() {
                    let xx = subset_columns(&proj_kn, cols.iter().cloned())?
                        .transpose()
                        .centre_columns()
                        .transpose();
                    assign_columns(&xx, cols.iter().cloned(), &mut proj_kn);
                }
            } else {
                warn!(
                    "row_weights projection: batch size {} != ncols {}",
                    col_to_batch.len(),
                    ncols
                );
            }
        }

        let (lb, ub) = (-4., 4.);
        proj_kn.scale_columns_inplace();
        if proj_kn.max() > ub || proj_kn.min() < lb {
            proj_kn.iter_mut().for_each(|x| {
                *x = x.clamp(lb, ub);
            });
            proj_kn.scale_columns_inplace();
        }
        Ok(RandColProjOut {
            basis: basis_dk,
            proj: proj_kn,
        })
    }

    /// Assign each column/cell to a sample by sorting through binary
    /// encoding
    ///
    /// # Arguments
    ///
    /// * `proj_kn` - random projection matrix (feature x column/cell)
    ///
    /// * `num_features` - number of top features (if None, use all of them)
    ///
    fn partition_columns_to_groups(
        &mut self,
        proj_kn: &nalgebra::DMatrix<f32>,
        num_sorting_features: Option<usize>,
        ncols_per_group: Option<usize>,
    ) -> anyhow::Result<usize> {
        let nn = proj_kn.ncols();
        if nn != self.num_columns() {
            return Err(anyhow::anyhow!("number of columns mismatch"));
        }

        let target_kk = num_sorting_features.unwrap_or(proj_kn.nrows());
        let kk = proj_kn.nrows().min(target_kk).min(nn);

        let binary_codes = binary_sort_columns(proj_kn, kk)?;
        let max_group = *binary_codes
            .iter()
            .max()
            .ok_or(anyhow::anyhow!("unable to determine max element"))?;
        self.assign_groups(&binary_codes, ncols_per_group);
        Ok(max_group + 1)
    }
}

/// Binarize the projection matrix and assign columns to some groups
///
/// # Arguments
/// * `proj_kn` - random projection matrix (feature x column/cell)
/// * `kk` - number of features
pub fn binary_sort_columns(
    proj_kn: &nalgebra::DMatrix<f32>,
    kk: usize,
) -> anyhow::Result<Vec<usize>> {
    // // standardization again to make sure that each coordinates is treated equally
    // let proj_nk = proj_kn.transpose().scale_columns();
    // let nn = proj_kn.ncols();
    // // SVD to spread out the points
    // let (mut q_nk, _, _) = proj_nk.rsvd(kk)?;
    // q_nk.scale_columns_inplace();

    // SVD to spread out the points
    let nn = proj_kn.ncols();
    let (_, _, mut q_nk) = proj_kn.rsvd(kk)?;
    q_nk.scale_columns_inplace();

    let mut binary_codes = DVector::<usize>::zeros(nn);
    for k in 0..kk {
        let binary_shift = |x: f32| -> usize {
            if x > 0.0 {
                1 << k
            } else {
                0
            }
        };
        binary_codes += q_nk.column(k).map(binary_shift);
    }

    Ok(binary_codes.data.as_vec().clone())
}

/// maximum binary code (binary shifting)
pub fn max_binary_code(kk: usize) -> usize {
    (1 << kk) - 1
}

#[cfg(test)]
mod repro_tests {
    use super::*;
    use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
    use std::sync::Arc;

    /// Small 8-gene × 12-cell `SparseIoVec` for exercising the projection.
    fn tiny_data_vec(dir: &tempfile::TempDir) -> anyhow::Result<SparseIoVec> {
        let path = dir.path().join("proj.zarr");
        let path_str = path.to_str().unwrap();
        let (d, n) = (8usize, 12usize);
        // Deterministic sparse pattern (nonzero where (i*7 + j*3) % 5 < 3).
        let mut triplets: Vec<(u64, u64, f32)> = Vec::new();
        for i in 0..d {
            for j in 0..n {
                if (i * 7 + j * 3) % 5 < 3 {
                    triplets.push((i as u64, j as u64, 1.0 + ((i + j) % 4) as f32));
                }
            }
        }
        let nnz = triplets.len();
        let mut backend = create_sparse_from_triplets(
            &triplets,
            (d, n, nnz),
            Some(path_str),
            Some(&SparseIoBackend::Zarr),
        )?;
        let gene_names: Vec<Box<str>> = (0..d).map(|i| format!("g{i}").into()).collect();
        let cell_names: Vec<Box<str>> = (0..n).map(|j| format!("c{j}").into()).collect();
        backend.register_row_names_vec(&gene_names);
        backend.register_column_names_vec(&cell_names);
        let mut data_vec = SparseIoVec::new();
        data_vec.push(Arc::from(backend), None)?;
        Ok(data_vec)
    }

    /// The seeded projection must be byte-identical across runs for a fixed seed
    /// — this is the reproducibility guarantee the gem/bge pipeline relies on.
    #[test]
    fn seeded_projection_is_reproducible() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let data_vec = tiny_data_vec(&dir)?;
        let a =
            data_vec.project_columns_with_batch_correction_seeded::<usize>(4, None, None, 123)?;
        let b =
            data_vec.project_columns_with_batch_correction_seeded::<usize>(4, None, None, 123)?;
        assert_eq!(a.basis, b.basis, "same seed → identical basis");
        assert_eq!(a.proj, b.proj, "same seed → identical projection");
        Ok(())
    }

    /// The non-seeded entry point must also be reproducible now (it pins
    /// `DEFAULT_PROJECTION_SEED` internally rather than drawing fresh entropy).
    #[test]
    fn default_projection_is_reproducible() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let data_vec = tiny_data_vec(&dir)?;
        let a = data_vec.project_columns(4, None)?;
        let b = data_vec.project_columns(4, None)?;
        assert_eq!(a.basis, b.basis);
        assert_eq!(a.proj, b.proj);
        Ok(())
    }

    /// Distinct seeds must yield distinct bases (the seed genuinely drives the
    /// projection, so `--seed` is authoritative where callers pass it).
    #[test]
    fn distinct_seeds_diverge() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let data_vec = tiny_data_vec(&dir)?;
        let a = data_vec.project_columns_with_batch_correction_seeded::<usize>(4, None, None, 1)?;
        let b = data_vec.project_columns_with_batch_correction_seeded::<usize>(4, None, None, 2)?;
        assert_ne!(a.basis, b.basis, "distinct seeds → distinct basis");
        Ok(())
    }
}
