use asap_data::sparse_matrix_hdf5::SparseMtxData;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use std::path::PathBuf;
use std::time::Instant;
use tempfile::tempdir;

fn create_temp_dir_file(_suffix: &str) -> anyhow::Result<PathBuf> {
    let _tempdir = tempdir()?.path().to_path_buf();
    std::fs::create_dir_all(&_tempdir)?;
    let _tempfile = tempfile::Builder::new()
        .suffix(&_suffix)
        .tempfile_in(_tempdir)?
        .path()
        .to_owned();

    if _tempfile.exists() {
        std::fs::remove_dir(&_tempfile)?;
    }

    Ok(_tempfile)
}

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
fn random_mtx_loading() -> anyhow::Result<()> {
    // 1. generate a random array2
    let a = Array::random((9, 1111), rand::distributions::Uniform::new(0., 1.));

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
        let b = measure_time(|| data.read_columns(3..4).unwrap());
        dbg!(&b);

        // 5. read the column 2
        let c = measure_time(|| data.read_columns(vec![3]).unwrap());
        dbg!(&c);

        // 6. open the backend file directly
        let backend_file = data.get_backend_file_name();
        let new_data = SparseMtxData::open(backend_file)?;
        let d = measure_time(|| new_data.read_columns(3..4).unwrap());
        dbg!(&d);

        let e = measure_time(|| new_data.read_columns(vec![7]).unwrap());
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
    let a = Array::random((5, 7), rand::distributions::Uniform::new(0., 1.));

    if let Ok(data) = SparseMtxData::from_ndarray(&a, None, None) {
        let a = a.select(Axis(1), &[2]);

        dbg!(&a);

        let b = data.read_columns(2..3).unwrap();

        dbg!(&b);

        let c = data.read_columns(vec![2]).unwrap();

        dbg!(&c);

        data.remove_backend_file()?;

        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    Ok(())
}
