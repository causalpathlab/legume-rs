
# Data Backend for Extraction And Neighbourhood Search

Here, we implement a wrapper for sparse matrix backends for swift data access by rows and columns without populating everything in memory. Being inspired by [`anndata-rs`](https://github.com/kaizhang/anndata-rs), we use `hdf5` and `zarr` as the back-end storage format. The `hdf5` and `zarr` file (or directory) is organized as follows:

```
(root)
    ├── nrow
    ├── ncol
    ├── nnz
    ├── by_column
    │   ├── data
    │   ├── indices (row indices)
    │   └── indptr (column pointers)
    └── by_row
        ├── data
        ├── indices (column indices)
        └── indptr (row pointers)
```
