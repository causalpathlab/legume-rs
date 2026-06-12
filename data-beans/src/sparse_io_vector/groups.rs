#![allow(dead_code)]

use super::*;

impl SparseIoVec {
    /// Assign columns to groups
    ///
    /// * `column_to_group` - column to group membership
    /// * `ncolumns_per_group` - number of columns per group. `None`: assign all the columns to the groups; `Some(x)`: limit the maximum number of columns per group to at most `x`.
    ///
    pub fn assign_groups<T>(&mut self, column_to_group: &[T], ncolumns_per_group: Option<usize>)
    where
        T: Sync + Send + std::hash::Hash + Eq + Clone + ToString,
    {
        let partitions = partition_by_membership(column_to_group, ncolumns_per_group);

        // Sort by keys to ensure consistent ordering
        let mut sorted_partitions: Vec<_> = partitions
            .into_iter()
            .map(|(k, cols)| (k.to_string().into_boxed_str(), cols))
            .collect();
        sorted_partitions.sort_by(|a, b| a.0.cmp(&b.0));

        let (group_keys, group_to_cols): (Vec<Box<str>>, Vec<Vec<usize>>) =
            sorted_partitions.into_iter().unzip();

        let col_to_group: HashMap<_, _> = group_to_cols
            .iter()
            .enumerate()
            .flat_map(|(g, cols)| cols.iter().map(move |&j| (j, g)))
            .collect();

        self.group_keys = Some(group_keys);
        self.group_to_cols = Some(group_to_cols);
        self.col_to_group = Some(col_to_group);
    }

    /// Take a vector of columns where each vector corresponds to a set
    pub fn take_grouped_columns(&self) -> Option<&Vec<Vec<usize>>> {
        self.group_to_cols.as_ref()
    }

    /// Get the group keys in the same order as group indices
    pub fn group_keys(&self) -> Option<&Vec<Box<str>>> {
        self.group_keys.as_ref()
    }

    /// Get a mapping from group keys to their column indices
    pub fn group_key_to_cols(&self) -> Option<HashMap<Box<str>, Vec<usize>>> {
        if let (Some(keys), Some(cols)) = (&self.group_keys, &self.group_to_cols) {
            Some(
                keys.iter()
                    .zip(cols.iter())
                    .map(|(k, c)| (k.clone(), c.clone()))
                    .collect(),
            )
        } else {
            None
        }
    }

    /// Take a vector of backend file and corresponding column indices
    pub fn take_backend_columns(&self) -> Vec<(Box<str>, Vec<usize>)> {
        self.data_to_cols
            .iter()
            .filter_map(|(&didx, cols)| {
                if let Some(arc_data) = self.data_vec.get(didx) {
                    let k = arc_data.get_backend_file_name();
                    // Drop sentinels left by `mask_columns` (masked-out cells).
                    let kept: Vec<usize> =
                        cols.iter().copied().filter(|&c| c != usize::MAX).collect();
                    Some((Box::<str>::from(k), kept))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Recall the `cells` group assignment; Note that this can be
    /// differ from the original vector used in `assign_groups` as we
    /// can have different number of columns and groups.
    pub fn get_group_membership<I>(&self, cells: I) -> anyhow::Result<Vec<usize>>
    where
        I: Iterator<Item = usize>,
    {
        let cell_to_group = self
            .col_to_group
            .as_ref()
            .expect("groups were not assigned");

        cells
            .map(|j| {
                cell_to_group
                    .get(&j)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("missing group membership"))
            })
            .collect()
    }

    /// number of groups
    pub fn num_groups(&self) -> usize {
        self.group_to_cols.as_ref().map(|x| x.len()).unwrap_or(0)
    }
}
