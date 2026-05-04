use crate::hdf5_io::read_hdf5_strings;
use clap::Args;
use log::info;
use matrix_util::common_io::file_ext;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct ListH5Args {
    pub h5_file: Box<str>,
}

#[derive(Args, Debug)]
pub struct ListZarrArgs {
    pub zarr_file: Box<str>,
}

/// List contents of a Zarr file (either .zarr directory or .zip archive)
pub fn list_zarr(cmd_args: &ListZarrArgs) -> anyhow::Result<()> {
    let ext = file_ext(&cmd_args.zarr_file)?;

    match ext.to_string().as_ref() {
        "zarr" => {
            info!(".zarr file: {}", cmd_args.zarr_file.as_ref());
            use zarrs::config::MetadataRetrieveVersion;
            use zarrs::filesystem::FilesystemStore;
            use zarrs::node::Node;

            let store = Arc::new(FilesystemStore::new(cmd_args.zarr_file.as_ref())?);
            let node = Node::open_opt(&store, "/", &MetadataRetrieveVersion::Default).unwrap();
            let tree = node.hierarchy_tree();
            println!("hierarchy_tree:\n{}", tree);
        }
        "zip" => {
            info!("zipped .zarr file: {}", cmd_args.zarr_file.as_ref());
            use zip::ZipArchive;

            let file = std::fs::File::open(cmd_args.zarr_file.as_ref())?;
            let reader = std::io::BufReader::new(file);

            let mut archive = ZipArchive::new(reader)?;

            for i in 0..archive.len() {
                let file = archive.by_index(i)?;
                println!("{}", file.name());
                // if file.is_dir() {
                //     println!("{}", file.name());
                // }
            }
        }
        _ => {
            info!("unknown extension '{}'", ext);
        }
    };

    Ok(())
}

/// List contents of an HDF5 file.
///
/// Prints the group/dataset hierarchy (with shapes for datasets) and, for
/// every dataset whose name suggests it holds 10x-style feature types
/// (`feature_type` / `feature_types`), prints a histogram of distinct values
/// so the caller knows what to pass to `from-h5 --select-row-type`.
pub fn list_h5(cmd_args: &ListH5Args) -> anyhow::Result<()> {
    let data_file = cmd_args.h5_file.clone();
    let file = hdf5::File::open(data_file.to_string())?;
    info!("Opened {}", data_file.clone());

    let mut feature_type_paths: Vec<String> = Vec::new();

    fn walk(
        group: &hdf5::Group,
        path: &str,
        indent: usize,
        feature_type_paths: &mut Vec<String>,
    ) -> hdf5::Result<()> {
        for member in group.member_names()? {
            let child_path = if path.is_empty() {
                member.clone()
            } else {
                format!("{}/{}", path, member)
            };

            if let Ok(subgroup) = group.group(&member) {
                println!("{:indent$}{}/", "", member, indent = indent);
                walk(&subgroup, &child_path, indent + 2, feature_type_paths)?;
            } else if let Ok(ds) = group.dataset(&member) {
                let shape = ds.shape();
                let dtype = ds
                    .dtype()
                    .ok()
                    .and_then(|d| d.to_descriptor().ok())
                    .map(|d| format!("{:?}", d))
                    .unwrap_or_else(|| "?".to_string());
                println!(
                    "{:indent$}{}  [{:?} {}]",
                    "",
                    member,
                    shape,
                    dtype,
                    indent = indent
                );

                if member == "feature_type" || member == "feature_types" {
                    feature_type_paths.push(child_path);
                }
            } else {
                println!("{:indent$}{}", "", member, indent = indent);
            }
        }
        Ok(())
    }

    walk(&file, "", 0, &mut feature_type_paths)?;

    if let Ok(shape_ds) = file.dataset("matrix/shape") {
        if let Ok(s) = shape_ds.read_1d::<i64>() {
            println!();
            println!("matrix/shape = {:?}  (features x barcodes)", s.to_vec());
        }
    }

    for ft_path in &feature_type_paths {
        println!();
        println!("feature types at /{}:", ft_path);
        match file
            .dataset(ft_path)
            .map_err(anyhow::Error::from)
            .and_then(read_hdf5_strings)
        {
            Ok(values) => {
                let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
                for v in &values {
                    *counts.entry(v.as_ref()).or_insert(0) += 1;
                }
                for (ty, n) in &counts {
                    println!("  {:>10}  {}", n, ty);
                }
                println!("  (total: {})", values.len());
            }
            Err(e) => {
                println!("  (failed to read: {})", e);
            }
        }
    }

    Ok(())
}
