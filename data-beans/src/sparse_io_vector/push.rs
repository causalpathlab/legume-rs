#![allow(dead_code)]

use super::*;

impl SparseIoVec {
    /// Add a backend's columns to the vector.
    ///
    /// * `data`: `Arc` to the backend [`SparseData`].
    /// * `data_name`: under [`ColumnAlignment::Disjoint`], appended as the
    ///   `@<data_name>` display disambiguator for barcodes shared across
    ///   files. **Ignored under [`ColumnAlignment::Union`]**, where cells
    ///   glue by raw barcode — use [`push_with_barcode_suffix`] to attach a
    ///   per-cell sample tag that participates in the Union merge.
    ///
    /// [`push_with_barcode_suffix`]: Self::push_with_barcode_suffix
    pub fn push(
        &mut self,
        data: Arc<SparseData>,
        data_name: Option<Box<str>>,
    ) -> anyhow::Result<()> {
        self.push_with_barcode_suffix(data, data_name, None)
    }

    /// Like [`push`](Self::push) but, under [`ColumnAlignment::Union`], tags
    /// every barcode of this backend with `{COLUMN_SEP}{barcode_suffix}`
    /// **before** the canonical-merge step. Two backends that share a barcode
    /// merge into one global cell only if they carry the SAME suffix, so
    /// callers can encode per-file sample identity (e.g. `rep1_wt`):
    /// same-sample modalities merge, different samples stay distinct. The
    /// tagged name also becomes the displayed column name. `None` reproduces
    /// [`push`](Self::push) exactly. The `data_name` Disjoint disambiguator is
    /// orthogonal: it still applies under `Disjoint`, and `barcode_suffix`
    /// only under `Union` (the two alignments are mutually exclusive).
    pub fn push_with_barcode_suffix(
        &mut self,
        data: Arc<SparseData>,
        data_name: Option<Box<str>>,
        barcode_suffix: Option<&str>,
    ) -> anyhow::Result<()> {
        let Some(ncol_data) = data.num_columns() else {
            return Err(anyhow::anyhow!("data file has no columns"));
        };
        debug_assert_eq!(self.col_to_data.len(), self.offset);
        let didx = self.data_vec.len();
        let didx_u32: u32 = didx
            .try_into()
            .map_err(|_| anyhow::anyhow!("backend count overflows u32"))?;
        let raw_col_names = data.column_names()?;
        debug_assert_eq!(raw_col_names.len(), ncol_data);

        // `Disjoint`: every push appends fresh global columns + the
        // `@<basename>` disambiguator. `Union`: match each pushed
        // barcode against the existing global cell pool by canonical
        // name so matched barcodes share a global column across
        // backends (reads merge their nonzeros into one output col).
        let data_to_cells_loc_to_glob: Vec<usize> = match self.column_alignment {
            ColumnAlignment::Disjoint => {
                let mut local_to_glob = Vec::with_capacity(ncol_data);
                for loc in 0..ncol_data {
                    let glob = self.offset + loc;
                    let loc_u32: u32 = loc
                        .try_into()
                        .map_err(|_| anyhow::anyhow!("local col overflows u32"))?;
                    self.col_to_data.push(vec![BackendLocation {
                        backend: didx_u32,
                        local_col: loc_u32,
                    }]);
                    local_to_glob.push(glob);
                }
                let data_tag = match data_name.as_deref() {
                    Some(x) => COLUMN_SEP.to_string() + x,
                    None => String::new(),
                };
                self.column_names_with_data_tag.extend(
                    raw_col_names
                        .iter()
                        .map(|x| (x.to_string() + &data_tag).into_boxed_str()),
                );
                self.offset += ncol_data;
                local_to_glob
            }
            ColumnAlignment::Union => {
                // First pass: detect within-backend duplicate barcodes
                // (would create a malformed global col with two entries
                // from the same backend).
                let mut seen_in_this_push: HashMap<Box<str>, usize> =
                    HashMap::with_capacity_and_hasher(ncol_data, Default::default());
                let mut local_to_glob = Vec::with_capacity(ncol_data);
                for (loc, raw) in raw_col_names.iter().enumerate() {
                    // Tag the barcode with the per-file suffix (sample id)
                    // before canonicalization so the merge key — and the
                    // displayed name — carry sample identity. Same suffix
                    // across backends ⇒ merge; different ⇒ distinct cells.
                    // Borrow `raw` in the common no-suffix path so the merge
                    // pass doesn't allocate a `Box<str>` per column.
                    let tagged: Cow<str> = match barcode_suffix {
                        Some(s) => Cow::Owned(format!("{raw}{COLUMN_SEP}{s}")),
                        None => Cow::Borrowed(raw.as_ref()),
                    };
                    let canon: Box<str> = match self.column_canonicalizer.as_ref() {
                        Some(c) => c(&tagged),
                        None => Box::from(&*tagged),
                    };
                    if let Some(&prev_loc) = seen_in_this_push.get(&canon) {
                        return Err(anyhow::anyhow!(
                            "ColumnAlignment::Union: backend {} has duplicate canonical \
                             barcode `{}` (local cols {} and {}) — Union cannot fold a \
                             cell with itself within one backend",
                            didx,
                            canon,
                            prev_loc,
                            loc,
                        ));
                    }
                    seen_in_this_push.insert(canon.clone(), loc);

                    let loc_u32: u32 = loc
                        .try_into()
                        .map_err(|_| anyhow::anyhow!("local col overflows u32"))?;
                    let glob = match self.col_name_position.get(&canon).copied() {
                        Some(g) => {
                            self.col_to_data[g as usize].push(BackendLocation {
                                backend: didx_u32,
                                local_col: loc_u32,
                            });
                            g as usize
                        }
                        None => {
                            let new_g = self.offset;
                            let new_g_u32: u32 = new_g
                                .try_into()
                                .map_err(|_| anyhow::anyhow!("global col overflows u32"))?;
                            self.col_name_position.insert(canon, new_g_u32);
                            self.col_to_data.push(vec![BackendLocation {
                                backend: didx_u32,
                                local_col: loc_u32,
                            }]);
                            // Tagged barcode (`raw` when no suffix) becomes
                            // the displayed name; subsequent backends
                            // contributing to this cell don't relabel it.
                            self.column_names_with_data_tag.push(Box::from(&*tagged));
                            self.offset += 1;
                            new_g
                        }
                    };
                    local_to_glob.push(glob);
                }
                local_to_glob
            }
        };

        // `data_to_cols[didx][loc] = glob`. Stored separately from
        // `col_to_data` because callers (e.g. `rows_triplets`) need the
        // forward map per backend.
        let entry = self.data_to_cols.entry(didx).or_default();
        entry.extend(data_to_cells_loc_to_glob.iter().copied());

        let row_names = data.row_names()?;
        let mut local_to_global = Vec::with_capacity(row_names.len());
        for row in row_names.iter() {
            let mut key: Box<str> = match self.row_canonicalizer.as_ref() {
                Some(canon) => canon(row),
                None => row.clone(),
            };
            // Per-backend modality namespacing: append `/{suffix}` after
            // canonicalization so the gene/locus rule still applies to the
            // bare name and only the modality tag distinguishes the row.
            if let Some(suffixes) = self.per_backend_row_suffix.as_ref() {
                let suffix = suffixes.get(didx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "per_backend_row_suffix has {} entries but backend index is {}",
                        suffixes.len(),
                        didx,
                    )
                })?;
                key = format!("{key}/{suffix}").into_boxed_str();
            }
            let glob_row = match self.row_name_position.get(&key) {
                Some(&g) => g,
                None => {
                    let next_global = self.row_names_by_global.len();
                    self.row_name_position.insert(key.clone(), next_global);
                    self.row_names_by_global.push(key);
                    self.row_count_by_global.push(0);
                    next_global
                }
            };
            local_to_global.push(glob_row);
        }
        // Count per-dataset *presence*, not per-local-row hits: when the
        // row canonicalizer collapses several local rows in one file to
        // the same global key, this dataset should still contribute
        // exactly 1 to that global's count. Otherwise the
        // `RowAlignment::Intersect` admit-rule (`count >= n_datasets`)
        // silently excludes every collapsed row.
        let mut counted: HashSet<usize> = HashSet::default();
        for &g in &local_to_global {
            if counted.insert(g) {
                self.row_count_by_global[g] += 1;
            }
        }
        // Flag canonicalizer-induced intra-file row merges so the read
        // paths can sum duplicate (row, col) entries instead of emitting
        // them twice (which breaks `from_nonzero_triplets` strictness).
        let has_intra_merges = counted.len() != local_to_global.len();
        self.data_has_intra_row_merges.push(has_intra_merges);
        let mut g2l: HashMap<usize, usize> =
            HashMap::with_capacity_and_hasher(local_to_global.len(), Default::default());
        for (l, &g) in local_to_global.iter().enumerate() {
            g2l.insert(g, l);
        }
        self.data_global_to_local_row.push(g2l);
        self.data_local_to_global_row.push(local_to_global);

        self.data_vec.push(data.clone());

        self.cached_num_columns = self.offset;
        self.recompute_row_mapping();

        info!(
            "Added {} columns ({} total); row {} = {}",
            ncol_data,
            self.offset,
            match self.row_alignment {
                RowAlignment::Intersect => "intersection",
                RowAlignment::Union => "union",
            },
            self.cached_num_rows
        );
        Ok(())
    }

    /// Recompute the global → compact row mapping under the current
    /// [`RowAlignment`] mode. Intersection keeps only rows present in
    /// every backend; Union keeps every row that any backend contains.
    /// Surviving rows get a compact index in raw-global order;
    /// everything else maps to `None`.
    fn recompute_row_mapping(&mut self) {
        let n_datasets = self.data_local_to_global_row.len();
        let n_global = self.row_names_by_global.len();
        let min_count = match self.row_alignment {
            RowAlignment::Intersect => n_datasets,
            RowAlignment::Union => 1,
        };

        self.global_to_compact_row.clear();
        self.global_to_compact_row.resize(n_global, None);
        self.compact_to_global_row.clear();

        let mut next_compact = 0usize;
        for g in 0..n_global {
            if self.row_count_by_global[g] >= min_count {
                self.global_to_compact_row[g] = Some(next_compact);
                self.compact_to_global_row.push(g);
                next_compact += 1;
            }
        }
        self.cached_num_rows = next_compact;
    }

    pub fn num_columns_by_data(&self) -> anyhow::Result<Vec<usize>> {
        Ok(self
            .data_vec
            .iter()
            .map(|d| d.num_columns().unwrap_or(0_usize))
            .collect())
    }

    pub fn remove_backend_file(&mut self) -> anyhow::Result<()> {
        for dat in self.data_vec.iter() {
            dat.remove_backend_file()?;
        }
        Ok(())
    }
}
