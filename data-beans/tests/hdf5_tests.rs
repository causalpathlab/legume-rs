use data_beans::simulate::*;
use data_beans::sparse_io::*;
use data_beans::sparse_matrix_hdf5::SparseMtxData;
use matrix_util::common_io::create_temp_dir_file;
use matrix_util::traits::SampleOps;

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

#[test]
fn random_ndarray_subset() -> anyhow::Result<()> {
    let whole_mat = Array2::<f32>::runif(17, 999);

    if let Ok(mut data) = SparseMtxData::from_ndarray(&whole_mat, None, None) {
        let nrow = data.num_rows().unwrap();
        let ncol = data.num_columns().unwrap();

        let rows: Vec<Box<str>> = (0..nrow).map(|x| x.to_string().into_boxed_str()).collect();
        let cols: Vec<Box<str>> = (0..ncol).map(|x| x.to_string().into_boxed_str()).collect();
        data.register_column_names_vec(&cols);
        data.register_row_names_vec(&rows);

        let a = whole_mat.select(Axis(1), &[9, 10, 500, 11, 1, 2, 3]);

        data.subset_columns_rows(Some(&vec![9, 10, 500, 11, 1, 2, 3]), None)?;

        let b = data.read_columns_ndarray((0..data.num_columns().unwrap()).collect())?;

        debug_assert_eq!(a, b);

        let a = a.select(Axis(0), &[1, 7, 16]);

        data.subset_columns_rows(None, Some(&vec![1, 7, 16]))?;
        let b = data.read_columns_ndarray((0..data.num_columns().unwrap()).collect())?;

        debug_assert_eq!(a, b);
        let a: Array2<f32> = a.select(Axis(0), &[2, 1, 0]);

        let ncol = a.shape()[1];
        let aa: Array2<f32> = Array2::zeros((1, ncol));

        dbg!(&a);
        dbg!(&aa);

        let c = ndarray::concatenate(Axis(0), &[a.view(), aa.view()])?;

        let new_row_names = vec!["16", "7", "1", "9"]
            .into_iter()
            .map(|x| x.to_string().into_boxed_str())
            .collect::<Vec<_>>();
        data.reorder_rows(&new_row_names)?;

        let b = data.read_columns_ndarray((0..data.num_columns().unwrap()).collect())?;

        debug_assert_eq!(b, c);
    }
    Ok(())
}

#[test]
fn random_mtx_loading() -> anyhow::Result<()> {
    // 1. generate a random array2
    let a = Array2::<f32>::runif(9, 1111);

    if let Ok(data) = SparseMtxData::from_ndarray(&a, None, None) {
        let a = a.select(Axis(1), &[3]);
        dbg!(&a);

        // 2. save it to mtx file
        let mtx_file = create_temp_dir_file(".mtx.gz")?;
        measure_time(|| data.to_mtx_file(mtx_file.to_str().unwrap()))?;

        // 3. create another data from the mtx file
        let data =
            measure_time(|| SparseMtxData::from_mtx_file(mtx_file.to_str().unwrap(), None, None));
        let data = data?;

        // 4. read the column 2
        let b = measure_time(|| data.read_columns_ndarray((3..4).collect()).unwrap());
        dbg!(&b);

        // 5. read the column 2
        let c = measure_time(|| data.read_columns_ndarray(vec![3]).unwrap());
        dbg!(&c);

        // 6. open the backend file directly
        let backend_file = data.get_backend_file_name();
        let new_data = SparseMtxData::open(backend_file)?;
        let d = measure_time(|| new_data.read_columns_ndarray((3..4).collect()).unwrap());
        dbg!(&d);

        let e = measure_time(|| new_data.read_columns_ndarray(vec![7]).unwrap());
        dbg!(&e);

        assert_ne!(a, e);

        // 7. remove the backend file
        data.remove_backend_file()?;
        new_data.remove_backend_file()?;

        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(c, d);
    }

    Ok(())
}

#[test]
fn random_ndarray_loading() -> anyhow::Result<()> {
    // let mut rng = rand::thread_rng();
    let a = Array2::<f32>::runif(5, 7);

    if let Ok(data) = SparseMtxData::from_ndarray(&a, None, None) {
        let a = a.select(Axis(1), &[2]);

        dbg!(&a);

        let b = data.read_columns_ndarray((2..3).collect()).unwrap();

        dbg!(&b);

        let c = data.read_columns_ndarray(vec![2]).unwrap();

        dbg!(&c);

        data.remove_backend_file()?;

        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    Ok(())
}

#[test]
fn simulate() -> anyhow::Result<()> {
    let sim_args = SimArgs {
        rows: 7,
        cols: 11,
        depth: 100,
        factors: 1,
        batches: 1,
        overdisp: 1.,
        pve_topic: 1.,
        pve_batch: 1.,
        rseed: 1,
        hierarchical_depth: None,
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

    let yy: Array2<f32> = data.read_columns_ndarray((0..n).collect())?;
    dbg!(&yy);

    let zz: Array2<f32> = data.read_rows_ndarray((0..m).collect())?;
    dbg!(&zz);

    data.remove_backend_file()?;

    if let Some(temp_dir) = Path::new(&mtx_file).parent() {
        std::fs::remove_dir_all(temp_dir)?;
    }

    Ok(())
}
