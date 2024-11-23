
Here, we implement a wrapper for `mmutil` traits for swift data access by rows and columns without populating everything in memory. Being inspired by [`anndata-rs`](https://github.com/kaizhang/anndata-rs), we use `hdf5` as the back-end storage format. The `hdf5` file is organized as follows:

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
