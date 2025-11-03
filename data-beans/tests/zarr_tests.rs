use data_beans::simulate::*;
use data_beans::sparse_io::*;
use data_beans::sparse_matrix_zarr::SparseMtxData;
use matrix_util::common_io::{create_temp_dir_file, read_lines};
use matrix_util::traits::SampleOps;

use approx::assert_abs_diff_eq;

use std::path::Path;
use std::time::Instant;

fn measure_time<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
    result
}

fn ndarray_to_dmatrix(array: &Array2<f32>) -> DMatrix<f32> {
    let (rows, cols) = array.dim();
    DMatrix::from_row_iterator(rows, cols, array.iter().cloned())
}

fn tensor_to_ndarray(tensor: Tensor) -> Array2<f32> {
    let shape = tensor.dims();
    let data = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    Array2::from_shape_vec((shape[0], shape[1]), data).unwrap()
}

#[test]
fn temp_array_zarrs() -> anyhow::Result<()> {
    use ndarray::prelude::*;
    use std::sync::Arc;
    // use zarrs::array::codec::ZstdCodec;
    // array_subset::ArraySubset,
    // storage::store,
    // let COMPRESSION_LEVEL = 5;
    // use std::fs::File;
    use zarrs::array::{DataType, FillValue};

    let backend_file = create_temp_dir_file(".zarr")?;
    let temp_filename = backend_file.to_str().expect("to_str failed");

    let store = Arc::new(zarrs::filesystem::FilesystemStore::new(temp_filename)?);

    // Create the root group
    zarrs::group::GroupBuilder::new()
        .build(store.clone(), "/")?
        .store_metadata()?;

    // Create an array
    let array_path = "/group/array";
    let array = zarrs::array::ArrayBuilder::new(
        vec![10, 10],                                // array shape
        vec![3_u64, 3_u64],                          // regular chunk shape
        DataType::Float32,                           // f32
        FillValue::from(zarrs::array::ZARR_NAN_F32), // nan
    )
    // .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(COMPRESSION_LEVEL, false))])
    .dimension_names(["y", "x"].into())
    .build(store.clone(), array_path)?;

    // Write array metadata to store
    array.store_metadata()?;

    println!(
        "Metadata:\n{}\n",
        serde_json::to_string_pretty(&array.metadata()).unwrap()
    );

    let subset_all = array.subset_all();
    let data_all = array.retrieve_array_subset_ndarray::<f32>(&subset_all)?;

    println!("ndarray::ArrayD:\n{data_all}");

    let chunk = array![[1_f32, 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let chunk = chunk.into_iter().collect::<Vec<_>>();

    array.store_chunk_elements(&[0, 0], &chunk)?;
    array.store_chunk_elements(&[0, 1], &chunk)?;
    array.store_chunk_elements(&[0, 2], &chunk)?;
    array.store_chunk_elements(&[0, 3], &chunk)?;

    let chunk = array![[1_f32, 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    array.store_array_subset_ndarray(&[1, 0], chunk)?;

    let data_all = array.retrieve_array_subset_ndarray::<f32>(&subset_all)?;
    println!("ndarray::ArrayD:\n{data_all}");

    Ok(())
}

#[test]
fn ndarray_dmatrix() -> anyhow::Result<()> {
    let raw_array = Array2::<f32>::runif(133, 373);

    let raw_matrix = ndarray_to_dmatrix(&raw_array);
    let mut data1 = create_sparse_from_ndarray(&raw_array, None, None)?;
    let mut data2 = create_sparse_from_dmatrix(&raw_matrix, None, None)?;

    data1.preload_columns()?;
    data2.preload_columns()?;

    {
        let a = data1.read_columns_ndarray((0..data1.num_columns().unwrap()).collect())?;
        let b = data2.read_columns_ndarray((0..data1.num_columns().unwrap()).collect())?;

        debug_assert_eq!(a, b);
    }

    {
        let a = data1.read_columns_dmatrix((0..data1.num_columns().unwrap()).collect())?;
        let b = data2.read_columns_dmatrix((0..data1.num_columns().unwrap()).collect())?;

        assert_abs_diff_eq!(&a, &b, epsilon = 1e-5);
    }

    data1.remove_backend_file()?;
    data2.remove_backend_file()?;
    Ok(())
}

#[test]
fn random_ndarray_subset() -> anyhow::Result<()> {
    let xx = Array2::<f32>::runif(333, 777);

    if let Ok(mut data) = SparseMtxData::from_ndarray(&xx, None, Some(true)) {
        let nrow = data.num_rows().unwrap();
        let ncol = data.num_columns().unwrap();

        let rows: Vec<Box<str>> = (0..nrow).map(|x| x.to_string().into_boxed_str()).collect();
        let cols: Vec<Box<str>> = (0..ncol).map(|x| x.to_string().into_boxed_str()).collect();
        data.register_column_names_vec(&cols);
        data.register_row_names_vec(&rows);
        data.print_hierarchy()?;

        data.preload_columns()?;

        {
            let cols = [9, 111, 11, 1, 2, 7, 3];
            let a = xx.select(Axis(1), &cols);

            let b = data.read_columns_tensor(Vec::from(&cols))?;
            debug_assert_eq!(a, tensor_to_ndarray(b));

            let c = data.read_columns_dmatrix(Vec::from(&cols))?;
            debug_assert_eq!(ndarray_to_dmatrix(&a), c);
        }

        {
            let rows = [9, 111, 11, 1, 2, 7, 3];
            let a = xx.select(Axis(0), &rows);

            let b = data.read_rows_tensor(Vec::from(&rows))?;
            debug_assert_eq!(a, tensor_to_ndarray(b));

            let c = data.read_rows_dmatrix(Vec::from(&rows))?;
            debug_assert_eq!(ndarray_to_dmatrix(&a), c);
        }

        // arbitrary subset and rearrange
        {
            let a = xx.select(Axis(1), &[9, 500, 10, 11, 1, 2, 3]);
            data.subset_columns_rows(Some(&vec![9, 500, 10, 11, 1, 2, 3]), None)?;
            data.preload_columns()?;

            let b = data.read_columns_ndarray((0..data.num_columns().unwrap()).collect())?;
            debug_assert_eq!(a, b);
            let a = xx.select(Axis(1), &[9, 500, 10, 11, 1, 2, 3]);
            let a = a.select(Axis(0), &[1, 7, 16]);
            let a: Array2<f32> = a.select(Axis(0), &[2, 1, 0]);
            let ncol = a.shape()[1];
            let aa: Array2<f32> = Array2::zeros((1, ncol));
            let c = ndarray::concatenate(Axis(0), &[a.view(), aa.view()])?;

            // check with this, also checking zero padding
            data.subset_columns_rows(None, Some(&vec![1, 7, 16]))?;
            data.preload_columns()?;

            let new_row_names = vec!["16", "7", "1", "9"]
                .into_iter()
                .map(|x| x.to_string().into_boxed_str())
                .collect::<Vec<_>>();
            data.reorder_rows(&new_row_names)?;
            data.preload_columns()?;

            let b = data.read_columns_ndarray((0..data.num_columns().unwrap()).collect())?;

            debug_assert_eq!(b, c);
        }
    }

    Ok(())
}

#[test]
fn simulate() -> anyhow::Result<()> {
    let sim_args = SimArgs {
        rows: 7,
        cols: 133,
        depth: 100,
        factors: 1,
        batches: 3,
        overdisp: 1.,
        pve_topic: 1.,
        pve_batch: 1.,
        rseed: 42,
    };

    let mtx_file = create_temp_dir_file(".mtx.gz")?;
    let mtx_file = mtx_file.to_str().unwrap().to_string();
    let dict_file = mtx_file.replace(".mtx.gz", ".dict.gz");
    let prop_file = mtx_file.replace(".mtx.gz", ".prop.gz");
    let memb_file = mtx_file.replace(".mtx.gz", ".memb.gz");
    let ln_batch_file = mtx_file.replace(".mtx.gz", ".ln_batch.gz");

    generate_factored_poisson_gamma_data_mtx(
        &sim_args,
        &mtx_file,
        &dict_file,
        &prop_file,
        &ln_batch_file,
        &memb_file,
    )?;

    let data = measure_time(|| SparseMtxData::from_mtx_file(&mtx_file, None, Some(true)));
    let data = data?;

    let n = data.num_columns().expect("failed to get #col") as usize;
    let m = data.num_rows().expect("failed to get #row") as usize;

    let batch_membership = read_lines(&memb_file)?
        .iter()
        .map(|x| x.parse::<usize>().unwrap())
        .collect::<Vec<usize>>();

    assert_eq!(batch_membership.len(), n);

    let _yy: Array2<f32> = data.read_columns_ndarray((0..n).collect())?;
    let _zz: Array2<f32> = data.read_rows_ndarray((0..m).collect())?;
    data.remove_backend_file()?;

    if let Some(temp_dir) = Path::new(&mtx_file).parent() {
        std::fs::remove_dir_all(temp_dir)?;
    }

    Ok(())
}

#[test]
fn random_mtx_loading() -> anyhow::Result<()> {
    // 1. generate a random array2
    let a = Array2::<f32>::runif(9, 1111);

    if let Ok(data) = SparseMtxData::from_ndarray(&a, None, Some(true)) {
        let a = a.select(Axis(1), &[7, 8, 9]);
        // dbg!(&a);

        // 2. save it to mtx file
        let mtx_file = create_temp_dir_file(".mtx.gz")?;
        measure_time(|| data.to_mtx_file(mtx_file.to_str().unwrap()))?;

        // 3. create another data from the mtx file
        let data = measure_time(|| {
            SparseMtxData::from_mtx_file(mtx_file.to_str().unwrap(), None, Some(true))
        });
        let data = data?;

        // 4. read the column 2
        let b = measure_time(|| data.read_columns_ndarray((7..10).collect()).unwrap());
        // dbg!(&b);

        // 6. open the backend file directly
        let backend_file = data.get_backend_file_name();
        let new_data = SparseMtxData::open(backend_file)?;
        let c = measure_time(|| new_data.read_columns_ndarray((7..10).collect()).unwrap());
        // dbg!(&c);

        // 7. remove the backend file
        data.remove_backend_file()?;
        new_data.remove_backend_file()?;

        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    Ok(())
}

#[test]
fn random_ndarray_loading() -> anyhow::Result<()> {
    let a = Array2::<f32>::runif(15, 7000);

    if let Ok(data) = SparseMtxData::from_ndarray(&a, None, None) {
        data.print_hierarchy()?;

        let a = a.select(Axis(1), &[2]);

        // dbg!(&a);

        let b = data.read_columns_ndarray((2..3).collect()).unwrap();

        // dbg!(&b);

        let c = data.read_columns_ndarray(vec![2]).unwrap();

        // dbg!(&c);

        data.remove_backend_file()?;

        assert_eq!(a, b);
        assert_eq!(b, c);
    } else {
        println!("Failed to create SparseMtxData from ndarray");
    }

    Ok(())
}
