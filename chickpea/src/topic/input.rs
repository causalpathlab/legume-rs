use crate::common::*;
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};

pub(crate) struct PairedDataWithBatch {
    pub data_stack: SparseIoStack,
    pub batch_membership: Vec<Box<str>>,
}

/// Load paired RNA + ATAC data, validate shared cells, return SparseIoStack.
///
/// `batch_files`, if provided, should have one entry per data file in
/// RNA-first order: `[rna_0, rna_1, ..., atac_0, atac_1, ...]`.
/// When `None`, each data file becomes its own batch (auto-detect).
pub fn load_paired_data(
    rna_files: &[Box<str>],
    atac_files: &[Box<str>],
    batch_files: Option<&[Box<str>]>,
) -> anyhow::Result<PairedDataWithBatch> {
    let n_rna = rna_files.len();
    let n_atac = atac_files.len();

    // Split batch files into RNA and ATAC portions
    let (rna_batch, atac_batch) = if let Some(bf) = batch_files {
        let expected = n_rna + n_atac;
        if bf.len() != expected {
            anyhow::bail!(
                "batch_files length {} != rna_files ({}) + atac_files ({})",
                bf.len(),
                n_rna,
                n_atac
            );
        }
        (Some(bf[..n_rna].to_vec()), Some(bf[n_rna..].to_vec()))
    } else {
        (None, None)
    };

    // Load RNA
    info!("Loading RNA data ({} file(s))...", n_rna);
    let rna = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: rna_files.to_vec(),
        batch_files: rna_batch,
        preload: false,
    })?;

    // Load ATAC
    info!("Loading ATAC data ({} file(s))...", n_atac);
    let atac = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: atac_files.to_vec(),
        batch_files: atac_batch,
        preload: false,
    })?;

    // Validate cell names match exactly
    validate_shared_cells(&rna.data, &atac.data)?;

    info!(
        "Loaded RNA: {} genes x {} cells, ATAC: {} peaks x {} cells",
        rna.data.num_rows(),
        rna.data.num_columns(),
        atac.data.num_rows(),
        atac.data.num_columns(),
    );

    // Build SparseIoStack: [0]=RNA, [1]=ATAC
    let mut data_stack = SparseIoStack::new();
    let batch_membership = rna.batch;
    data_stack.push(rna.data)?;
    data_stack.push(atac.data)?;

    Ok(PairedDataWithBatch {
        data_stack,
        batch_membership,
    })
}

fn validate_shared_cells(rna: &SparseIoVec, atac: &SparseIoVec) -> anyhow::Result<()> {
    let rna_names = rna.column_names()?;
    let atac_names = atac.column_names()?;

    if rna_names.len() != atac_names.len() {
        anyhow::bail!(
            "Cell count mismatch: RNA has {} cells, ATAC has {}",
            rna_names.len(),
            atac_names.len(),
        );
    }

    if rna_names != atac_names {
        // Count shared cells for a more informative error
        let rna_set: rustc_hash::FxHashSet<&str> = rna_names.iter().map(|s| s.as_ref()).collect();
        let n_shared = atac_names
            .iter()
            .filter(|s| rna_set.contains(s.as_ref()))
            .count();
        anyhow::bail!(
            "Cell name mismatch between RNA ({} cells) and ATAC ({} cells); \
             {} shared. Paired multiome data must have identical cell barcodes \
             in the same order.",
            rna_names.len(),
            atac_names.len(),
            n_shared,
        );
    }

    Ok(())
}
