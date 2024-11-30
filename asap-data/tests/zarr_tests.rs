use asap_data::common_io::read_lines;
use asap_data::simulate::*;
use asap_data::sparse_matrix_zarr::SparseMtxData;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use std::path::Path;
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
fn simulate() -> anyhow::Result<()> {
    let args = SimulateArgs {
        rows: 7,
        cols: 11,
        factors: None,
        batches: None,
        rseed: None,
    };

    let mtx_file = create_temp_dir_file(".mtx.gz")?;
    let mtx_file = mtx_file.to_str().unwrap().to_string();
    let dict_file = mtx_file.replace(".mtx.gz", ".dict.gz");
    let prop_file = mtx_file.replace(".mtx.gz", ".prop.gz");
    let memb_file = mtx_file.replace(".mtx.gz", ".memb.gz");
    let ln_batch_file = mtx_file.replace(".mtx.gz", ".ln_batch.gz");

    generate_factored_gamma_data_mtx(
        &args,
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

    let yy: Array2<f32> = data.read_columns(0..n)?;
    dbg!(&yy);

    let zz: Array2<f32> = data.read_rows(0..m)?;
    dbg!(&zz);

    data.remove_backend_file()?;

    if let Some(temp_dir) = Path::new(&mtx_file).parent() {
        std::fs::remove_dir_all(temp_dir)?;
    }

    Ok(())
}

#[test]
fn random_mtx_loading() -> anyhow::Result<()> {
    // 1. generate a random array2
    let a = Array::random((9, 1111), rand::distributions::Uniform::new(0., 1.));

    if let Ok(data) = SparseMtxData::from_ndarray(&a, None, Some(true)) {
        let a = a.select(Axis(1), &[7, 8, 9]);
        dbg!(&a);

        // 2. save it to mtx file
        let mtx_file = create_temp_dir_file(".mtx.gz")?;
        measure_time(|| data.to_mtx_file(mtx_file.to_str().unwrap()))?;

        // 3. create another data from the mtx file
        let data = measure_time(|| {
            SparseMtxData::from_mtx_file(mtx_file.to_str().unwrap(), None, Some(true))
        });
        let data = data?;

        // 4. read the column 2
        let b = measure_time(|| data.read_columns(7..10).unwrap());
        dbg!(&b);

        // 6. open the backend file directly
        let backend_file = data.get_backend_file_name();
        let new_data = SparseMtxData::open(backend_file)?;
        let c = measure_time(|| new_data.read_columns(7..10).unwrap());
        dbg!(&c);

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
    let a = Array::random((15, 7000), rand::distributions::Uniform::new(0., 1.));

    if let Ok(data) = SparseMtxData::from_ndarray(&a, None, None) {
        data.print_hierarchy()?;

        let a = a.select(Axis(1), &[2]);

        dbg!(&a);

        let b = data.read_columns(2..3).unwrap();

        dbg!(&b);

        let c = data.read_columns(vec![2]).unwrap();

        dbg!(&c);

        data.remove_backend_file()?;

        assert_eq!(a, b);
        assert_eq!(b, c);
    } else {
        println!("Failed to create SparseMtxData from ndarray");
    }

    Ok(())
}
