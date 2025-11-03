// #![allow(dead_code)]

use crate::sparse_io_vector::*;
use log::info;

/// `sparse_io_stack` is a stack of `sparse_io_vector`
pub struct SparseIoStack {
    pub data_stack: Vec<SparseIoVec>,
    column_names: Vec<Box<str>>,
}

impl Default for SparseIoStack {
    /// an empty sparse io vector for horizontal data integration
    fn default() -> Self {
        Self::new()
    }
}

impl SparseIoStack {
    pub fn new() -> Self {
        Self {
            data_stack: vec![],
            column_names: vec![],
        }
    }

    pub fn push(&mut self, data: SparseIoVec) -> anyhow::Result<()> {
        if self.data_stack.is_empty() {
            self.column_names.extend(data.column_names()?);
            self.data_stack.push(data);
            return Ok(());
        }

        info!("Checking column names...");
        if self.column_names != data.column_names()? {
            return Err(anyhow::anyhow!("column names don't match"));
        }
        self.data_stack.push(data);
        Ok(())
    }

    pub fn num_types(&self) -> usize {
        self.data_stack.len()
    }

    pub fn num_columns(&self) -> anyhow::Result<usize> {
        self.data_stack
            .iter()
            .map(|x| x.num_columns())
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .max()
            .ok_or(anyhow::anyhow!("can't figure out the max"))
    }
}
