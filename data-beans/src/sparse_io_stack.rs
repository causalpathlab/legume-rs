// #![allow(dead_code)]

use crate::sparse_io_vector::*;
use log::info;

/// `sparse_io_stack` is a stack of `sparse_io_vector`
pub struct SparseIoStack {
    pub stack: Vec<SparseIoVec>,
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
            stack: vec![],
            column_names: vec![],
        }
    }

    pub fn push(&mut self, data: SparseIoVec) -> anyhow::Result<()> {
        if self.stack.is_empty() {
            self.column_names.extend(data.column_names()?);
            self.stack.push(data);
            return Ok(());
        }

        info!("Checking column names...");
        if self.column_names != data.column_names()? {
            return Err(anyhow::anyhow!("column names don't match"));
        }
        self.stack.push(data);
        Ok(())
    }

    /// number of data types
    pub fn num_types(&self) -> usize {
        self.stack.len()
    }

    /// number of shared columns
    pub fn num_columns(&self) -> anyhow::Result<usize> {
        self.stack
            .iter()
            .map(|x| x.num_columns())
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .max()
            .ok_or(anyhow::anyhow!("can't figure out the max"))
    }

    /// Get the shared column names.
    pub fn column_names(&self) -> anyhow::Result<Vec<Box<str>>> {
        if self.column_names.len() != self.num_columns()? {
            return Err(anyhow::anyhow!("inconsistent columns"));
        }
        Ok(self.column_names.clone())
    }

    /// Get the row names combined across all the types. We will
    /// append additional data type index: `format!({}_{}, x, d)`
    ///
    pub fn row_names(&self) -> anyhow::Result<Vec<Box<str>>> {
        Ok(self
            .stack
            .iter()
            .enumerate()
            .map(|(d, x)| {
                x.row_names().map(|x| {
                    x.into_iter()
                        .map(|y| format!("{}_{}", y, d).into_boxed_str())
                        .collect::<Vec<_>>()
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect())
    }
}
