use matrix_util::common_io::file_ext;
use log::info;
use std::sync::Arc;

// Import Args from main.rs
use crate::{ListH5Args, ListZarrArgs};

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

/// List contents of an HDF5 file
pub fn list_h5(cmd_args: &ListH5Args) -> anyhow::Result<()> {
    let data_file = cmd_args.h5_file.clone();
    let file = hdf5::File::open(data_file.to_string())?;
    info!("Opened {}", data_file.clone());

    fn list_group(group: &hdf5::Group, indent: usize) -> hdf5::Result<()> {
        for member in group.member_names()? {
            println!("{:indent$}{}", "", member, indent = indent);

            if let Ok(obj) = group.group(&member) {
                if let Ok(subgroup) = obj.as_group() {
                    list_group(&subgroup, indent + 2)?;
                }
            }
        }
        Ok(())
    }

    list_group(&file, 0)?;

    Ok(())
}
