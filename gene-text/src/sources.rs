//! Where description text comes from. Every source reduces to one row per
//! feature: `(feature, type, name, text)`, the same four columns
//! `senna fne --export-text` writes, so a run can mix them freely.
//!
//! Sources and their go-to downloads:
//! - **UniProt** reviewed human proteins as one TSV (symbol, protein name,
//!   function): `https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true&fields=gene_primary,gene_synonym,protein_name,cc_function&format=tsv`
//! - **NCBI gene_info** (symbol, synonyms, full name only):
//!   `https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz`
//! - **OBO** ontologies with definitions: `http://purl.obolibrary.org/obo/go/go-basic.obo`,
//!   `http://purl.obolibrary.org/obo/cl/cl-basic.obo`
//! - **GMT** gene sets (MSigDB); the description column, or the set name
//!   when the description is a URL.
//! - **Generic** `feature <TAB> type <TAB> name <TAB> text` with that header.

use anyhow::{Context, Result};
use auxiliary_data::feature_names::FeatureNameKind;
use auxiliary_data::gene_sets::read_gmt;
use auxiliary_data::ontology::Ontology;
use log::{info, warn};
use matrix_util::common_io::open_buf_reader;
use rustc_hash::FxHashMap;
use std::io::BufRead;

pub const GENE_TYPE: &str = "gene";
pub const TERM_TYPE: &str = "term";

/// One feature's text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Doc {
    pub feature: Box<str>,
    pub ty: Box<str>,
    pub name: Box<str>,
    pub text: Box<str>,
}

impl Doc {
    /// What the encoder sees: the name, then the description, so a symbol
    /// with an empty description still carries its full name.
    #[must_use]
    pub fn sentence(&self) -> String {
        match (self.name.is_empty(), self.text.is_empty()) {
            (false, false) => format!("{}. {}", self.name, self.text),
            (false, true) => self.name.to_string(),
            (true, _) => self.text.to_string(),
        }
    }
}

/// Docs keyed by `(type, feature)`; a later source fills only what an
/// earlier one left empty.
#[derive(Default)]
pub struct Corpus {
    docs: Vec<Doc>,
    index: FxHashMap<(Box<str>, Box<str>), usize>,
    name_kind: FeatureNameKind,
}

impl Corpus {
    #[must_use]
    pub fn new(name_kind: FeatureNameKind) -> Self {
        Self {
            name_kind,
            ..Default::default()
        }
    }

    pub fn push(&mut self, ty: &str, feature: &str, name: &str, text: &str) {
        let feature: Box<str> = if ty == GENE_TYPE {
            self.name_kind.canonicalize(feature.trim())
        } else {
            feature.trim().into()
        };
        if feature.is_empty() {
            return;
        }
        let name = clean(name);
        let text = clean(text);
        let key = (Box::from(ty), feature.clone());
        match self.index.get(&key) {
            Some(&i) => {
                let d = &mut self.docs[i];
                if d.name.is_empty() {
                    d.name = name.into();
                }
                if d.text.is_empty() {
                    d.text = text.into();
                }
            }
            None => {
                self.index.insert(key, self.docs.len());
                self.docs.push(Doc {
                    feature,
                    ty: ty.into(),
                    name: name.into(),
                    text: text.into(),
                });
            }
        }
    }

    #[must_use]
    pub fn docs(&self) -> &[Doc] {
        &self.docs
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Drop features whose name and text are both empty.
    pub fn retain_with_text(&mut self) -> usize {
        let before = self.docs.len();
        self.docs
            .retain(|d| !d.name.is_empty() || !d.text.is_empty());
        self.index = self
            .docs
            .iter()
            .enumerate()
            .map(|(i, d)| ((d.ty.clone(), d.feature.clone()), i))
            .collect();
        before - self.docs.len()
    }

    /// `feature <TAB> type <TAB> name <TAB> text` with a header (what
    /// `senna fne --export-text` writes).
    pub fn add_generic_tsv(&mut self, path: &str) -> Result<()> {
        let reader = open_buf_reader(path).with_context(|| format!("opening {path}"))?;
        let mut n = 0usize;
        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = line.split('\t').collect();
            if i == 0
                && cols
                    .first()
                    .is_some_and(|c| c.eq_ignore_ascii_case("feature"))
            {
                continue;
            }
            if cols.len() < 2 {
                continue;
            }
            self.push(
                cols[1],
                cols[0],
                cols.get(2).copied().unwrap_or(""),
                cols.get(3).copied().unwrap_or(""),
            );
            n += 1;
        }
        info!("{path}: {n} text rows");
        Ok(())
    }

    /// UniProt REST TSV with the columns `Gene Names (primary)`,
    /// `Gene Names (synonym)` (optional), `Protein names`, `Function [CC]`
    /// (header-matched, any order).
    pub fn add_uniprot_tsv(&mut self, path: &str) -> Result<()> {
        let reader = open_buf_reader(path).with_context(|| format!("opening {path}"))?;
        let mut lines = reader.lines();
        let header = lines.next().context("empty UniProt TSV")??;
        let cols: Vec<&str> = header.split('\t').collect();
        let find = |names: &[&str]| -> Option<usize> {
            cols.iter().position(|c| {
                let c = c.trim().to_lowercase();
                names.iter().any(|n| c == *n)
            })
        };
        let i_gene = find(&["gene names (primary)", "gene_primary", "gene names"])
            .context("UniProt TSV: no `Gene Names (primary)` column")?;
        let i_protein = find(&["protein names", "protein_name"]);
        let i_function = find(&["function [cc]", "cc_function", "function"]);
        anyhow::ensure!(
            i_protein.is_some() || i_function.is_some(),
            "UniProt TSV: neither `Protein names` nor `Function [CC]` is present"
        );
        let mut n = 0usize;
        for line in lines {
            let line = line?;
            let f: Vec<&str> = line.split('\t').collect();
            let Some(genes) = f.get(i_gene) else { continue };
            let protein = i_protein.and_then(|i| f.get(i)).copied().unwrap_or("");
            let function = i_function
                .and_then(|i| f.get(i))
                .map(|s| strip_uniprot_function(s))
                .unwrap_or_default();
            // Several primary symbols are `; `-separated; each gets the text.
            for g in genes.split(';').map(str::trim).filter(|g| !g.is_empty()) {
                self.push(GENE_TYPE, g, protein, &function);
                n += 1;
            }
        }
        info!("{path}: {n} UniProt gene rows");
        Ok(())
    }

    /// NCBI `gene_info` (tab-delimited, `#tax_id` header): symbol (col 3),
    /// synonyms (col 5) and full name (col 9).
    pub fn add_ncbi_gene_info(&mut self, path: &str) -> Result<()> {
        let reader = open_buf_reader(path).with_context(|| format!("opening {path}"))?;
        let mut n = 0usize;
        for line in reader.lines() {
            let line = line?;
            if line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 9 {
                continue;
            }
            let synonyms = if f[4] == "-" { "" } else { f[4] };
            let text = if synonyms.is_empty() {
                String::new()
            } else {
                format!("Also known as {}.", synonyms.replace('|', ", "))
            };
            self.push(GENE_TYPE, f[2], f[8], &text);
            n += 1;
        }
        info!("{path}: {n} gene_info rows");
        Ok(())
    }

    /// Every term of an OBO ontology with its name and definition.
    pub fn add_obo(&mut self, path: &str) -> Result<()> {
        let onto = Ontology::load_obo(path)?;
        let mut n = 0usize;
        for id in onto.ids() {
            let name = onto.name(id).unwrap_or("");
            let def = onto.def(id).unwrap_or("");
            if name.is_empty() && def.is_empty() {
                continue;
            }
            self.push(TERM_TYPE, id, name, def);
            n += 1;
        }
        info!("{path}: {n} ontology terms with text");
        Ok(())
    }

    /// GMT sets: the description column as text, or nothing when it is a
    /// URL (MSigDB), in which case the set name is all there is.
    pub fn add_gmt(&mut self, path: &str) -> Result<()> {
        let sets = read_gmt(path)?;
        let mut n = 0usize;
        for term in sets.term_genes.keys() {
            let desc = sets.names.get(term).map(|d| d.as_ref()).unwrap_or("");
            let desc = if desc.starts_with("http://") || desc.starts_with("https://") {
                ""
            } else {
                desc
            };
            self.push(TERM_TYPE, term, &term.replace('_', " "), desc);
            n += 1;
        }
        info!("{path}: {n} gene sets");
        Ok(())
    }

    pub fn log_summary(&self) {
        let mut per_type: FxHashMap<&str, (usize, usize)> = FxHashMap::default();
        for d in &self.docs {
            let e = per_type.entry(&d.ty).or_default();
            e.0 += 1;
            if !d.text.is_empty() {
                e.1 += 1;
            }
        }
        let mut types: Vec<_> = per_type.into_iter().collect();
        types.sort();
        for (t, (n, with_text)) in types {
            info!("corpus: {n} `{t}` features, {with_text} with a description beyond the name");
        }
        if self.docs.is_empty() {
            warn!("corpus: no features with text");
        }
    }
}

/// UniProt's `FUNCTION: ... {ECO:...}. ...` → plain sentences.
fn strip_uniprot_function(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0usize;
    for ch in s.chars() {
        match ch {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    let out = out.replace("FUNCTION: ", "");
    clean(&out)
}

/// Collapse whitespace (tabs and newlines included) to single spaces.
fn clean(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
#[path = "sources_tests.rs"]
mod sources_tests;
