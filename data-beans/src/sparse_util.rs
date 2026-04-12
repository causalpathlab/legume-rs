use clap::ValueEnum;

pub struct ValuesIndicesPointers<'a> {
    pub values: &'a [f32],
    pub indices: &'a [u64],
    pub indptr: &'a [u64],
}

pub struct CooTripletsShape {
    pub triplets: Vec<(u64, u64, f32)>,
    pub shape: TripletsShape,
}

pub struct TripletsShape {
    pub nrows: usize,
    pub ncols: usize,
    pub nnz: usize,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum IndexPointerType {
    Column,
    Row,
}

pub trait SparseTripletsTraits {
    /// convert sparse pointers into (row, column, value) triplets
    fn to_coo(&self, pointer_type: IndexPointerType) -> anyhow::Result<CooTripletsShape>;
}

/////////////////////
// implementations //
/////////////////////

impl<'a> SparseTripletsTraits for ValuesIndicesPointers<'a> {
    fn to_coo(&self, pointer_type: IndexPointerType) -> anyhow::Result<CooTripletsShape> {
        use rayon::prelude::*;

        let indices = self.indices;
        let indptr = self.indptr;
        let values = self.values;

        let nelem = values.len();
        if nelem != indices.len() {
            return Err(anyhow::anyhow!(
                "`values` and `indices` have different sizes"
            ));
        }
        if indptr.is_empty() {
            return Err(anyhow::anyhow!("`indptr` is empty"));
        }
        let nvectors = indptr.len() - 1;

        // Build the triplet list in parallel without a global lock. `flat_map_iter`
        // lets each rayon worker produce a contiguous run of triplets from one
        // compressed vector, and rayon's ordered collect joins them into a single
        // Vec — no Arc<Mutex>, no per-vector intermediate Vec, no lock contention.
        let triplets: Vec<(u64, u64, f32)> = (0..nvectors)
            .into_par_iter()
            .flat_map_iter(|idx| {
                let j = idx as u64;
                let start = indptr[idx] as usize;
                let end = indptr[idx + 1] as usize;
                let end = end.min(nelem);
                let start = start.min(end);
                indices[start..end]
                    .iter()
                    .zip(values[start..end].iter())
                    .map(move |(&i, &x_ij)| match pointer_type {
                        IndexPointerType::Column => (i, j, x_ij),
                        IndexPointerType::Row => (j, i, x_ij),
                    })
            })
            .collect();

        let nnz = triplets.len();

        // Shape is derivable from the compressed layout directly — no need to
        // scan the whole triplets Vec again.
        let max_idx_index = indices.par_iter().copied().max().unwrap_or(0) as usize + 1;
        let (nrows, ncols) = match pointer_type {
            IndexPointerType::Column => (max_idx_index, nvectors),
            IndexPointerType::Row => (nvectors, max_idx_index),
        };

        Ok(CooTripletsShape {
            triplets,
            shape: TripletsShape { nrows, ncols, nnz },
        })
    }
}
