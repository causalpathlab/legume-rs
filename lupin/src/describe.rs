//! `lupin describe` — compose a short sentence from annotate / lineage_annot evidence.
//!
//! The composer **never decides**: evidence comes from annotate artifacts (and optional
//! keyword incidence from a `word-graph` vocab). Sentences are citation-checked against
//! the allowed entity set. Candle Hub decoding is not wired yet — templates only.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use clap::Args;
use gene_text::vocab::Vocabulary;
use log::info;
use matrix_util::parquet::peek_parquet_field_names;
use serde_json::json;

#[derive(Args, Debug)]
pub struct DescribeArgs {
    #[arg(
        long,
        short = 'f',
        help = "Annotate / lineage output prefix (reads `{from}.annot.parquet` or `{from}.argmax.tsv`)"
    )]
    pub from: Box<str>,

    #[arg(
        long,
        help = "Optional `word-graph` / `text-qc` prefix; attaches kept words from `{prefix}.vocab.tsv`"
    )]
    pub text_prefix: Option<Box<str>>,

    #[arg(long, short = 'o', help = "Output prefix (default: `--from`)")]
    pub out: Option<Box<str>>,
}

#[derive(Debug, Clone)]
struct ClusterEvidence {
    id: String,
    coarse_label: String,
    best_label: String,
    best_q: Option<f32>,
    best_significant: bool,
    incidence_words: Vec<String>,
}

pub fn run_describe(args: &DescribeArgs) -> Result<()> {
    let out = args.out.as_deref().unwrap_or(args.from.as_ref());
    matrix_util::common_io::mkdir_parent(out)?;

    let mut enriched = load_evidence(args.from.as_ref())?;
    let incidence = load_incidence_words(args.text_prefix.as_deref())?;
    if !incidence.is_empty() {
        for c in &mut enriched {
            c.incidence_words = incidence.iter().take(8).cloned().collect();
        }
    }

    let allowed = allowed_entities(&enriched);
    let mut sentences = Vec::new();
    for c in &enriched {
        let draft = template_sentence(c);
        let sentence = citation_check(&draft, &allowed, c)?;
        sentences.push((c.id.clone(), sentence));
    }

    let evidence = json!({
        "from": args.from.as_ref(),
        "text_prefix": args.text_prefix.as_ref().map(|s| s.as_ref()),
        "clusters": enriched.iter().map(|c| json!({
            "id": c.id,
            "coarse_label": c.coarse_label,
            "best_label": c.best_label,
            "best_q": c.best_q,
            "best_significant": c.best_significant,
            "incidence_words": c.incidence_words,
        })).collect::<Vec<_>>(),
        "allowed_entities": allowed.iter().cloned().collect::<Vec<_>>(),
    });

    let json_path = format!("{out}.describe.json");
    let mut jw = File::create(&json_path).with_context(|| format!("create {json_path}"))?;
    serde_json::to_writer_pretty(&mut jw, &evidence)?;
    writeln!(jw)?;
    info!("wrote {json_path}");

    let txt_path = format!("{out}.describe.txt");
    let mut tw = File::create(&txt_path).with_context(|| format!("create {txt_path}"))?;
    for (id, s) in &sentences {
        writeln!(tw, "{id}\t{s}")?;
    }
    info!("wrote {txt_path} ({} sentence(s))", sentences.len());
    Ok(())
}

fn load_evidence(from: &str) -> Result<Vec<ClusterEvidence>> {
    let annot = format!("{from}.annot.parquet");
    let argmax = format!("{from}.argmax.tsv");
    let lineage_annot = format!("{from}.lineage_annot.annot.parquet");

    if Path::new(&annot).exists() {
        return load_from_annot_parquet(&annot);
    }
    if Path::new(&lineage_annot).exists() {
        return load_from_annot_parquet(&lineage_annot);
    }
    if Path::new(&argmax).exists() {
        return load_from_argmax_tsv(&argmax);
    }
    bail!(
        "no annotate artifacts under `{from}` \
         (expected `{from}.annot.parquet`, `{from}.lineage_annot.annot.parquet`, or `{from}.argmax.tsv`)"
    );
}

fn load_from_annot_parquet(path: &str) -> Result<Vec<ClusterEvidence>> {
    let fields = peek_parquet_field_names(path).with_context(|| format!("peek {path}"))?;
    let has = |name: &str| fields.iter().any(|f| f.as_ref() == name);
    anyhow::ensure!(
        has("coarse_label") && has("community"),
        "{path}: need coarse_label and community columns"
    );

    let string_cols: Vec<&str> = if has("best_label") {
        vec!["coarse_label", "best_label"]
    } else {
        vec!["coarse_label"]
    };
    let mut numeric_cols: Vec<&str> = vec!["community"];
    if has("best_q") {
        numeric_cols.push("best_q");
    }
    if has("best_significant") {
        numeric_cols.push("best_significant");
    }

    let (strings, nums) =
        matrix_util::parquet::read_table_columns(path, &string_cols, &numeric_cols)
            .with_context(|| format!("read {path}"))?;

    let coarse = &strings[0];
    let best = if strings.len() > 1 {
        &strings[1]
    } else {
        coarse
    };
    let community: Vec<i32> = nums[0].iter().map(|x| *x as i32).collect();
    let best_q_col = numeric_cols.iter().position(|&c| c == "best_q");
    let best_sig_col = numeric_cols.iter().position(|&c| c == "best_significant");

    let mut by_comm: BTreeMap<i32, ClusterEvidence> = BTreeMap::new();
    for i in 0..community.len() {
        let id = community[i];
        by_comm.entry(id).or_insert_with(|| {
            let coarse_s = coarse[i].to_string();
            let best_s = best[i].to_string();
            ClusterEvidence {
                id: format!("K{id}"),
                coarse_label: coarse_s.clone(),
                best_label: best_s,
                best_q: best_q_col.map(|j| nums[j][i] as f32),
                best_significant: best_sig_col
                    .map(|j| nums[j][i] as i32 != 0)
                    .unwrap_or(coarse_s != "unassigned"),
                incidence_words: Vec::new(),
            }
        });
    }
    Ok(by_comm.into_values().collect())
}

fn load_from_argmax_tsv(path: &str) -> Result<Vec<ClusterEvidence>> {
    let f = File::open(path).with_context(|| format!("open {path}"))?;
    let mut lines = BufReader::new(f).lines();
    let _header = lines.next().transpose()?;
    let mut labels: BTreeSet<String> = BTreeSet::new();
    for line in lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split('\t');
        let _cell = parts.next();
        if let Some(lab) = parts.next() {
            labels.insert(lab.to_string());
        }
    }
    Ok(labels
        .into_iter()
        .enumerate()
        .map(|(i, lab)| {
            let sig = lab != "unassigned";
            ClusterEvidence {
                id: format!("L{i}"),
                coarse_label: lab.clone(),
                best_label: lab,
                best_q: None,
                best_significant: sig,
                incidence_words: Vec::new(),
            }
        })
        .collect())
}

fn load_incidence_words(prefix: Option<&str>) -> Result<Vec<String>> {
    let Some(prefix) = prefix else {
        return Ok(Vec::new());
    };
    let path = format!("{prefix}.vocab.tsv");
    if !Path::new(&path).exists() {
        info!("describe: no {path}; skipping keyword incidence");
        return Ok(Vec::new());
    }
    // n_docs is only used for df_frac display in Vocabulary; 1 is fine for incidence listing.
    let vocab = Vocabulary::read_tsv(&path, 1)?;
    Ok(vocab.kept.iter().take(32).map(|w| w.to_string()).collect())
}

fn allowed_entities(clusters: &[ClusterEvidence]) -> BTreeSet<String> {
    let mut s = BTreeSet::new();
    for c in clusters {
        if c.coarse_label != "unassigned" {
            s.insert(c.coarse_label.clone());
        }
        if c.best_label != "unassigned" {
            s.insert(c.best_label.clone());
        }
        for w in &c.incidence_words {
            s.insert(w.clone());
        }
    }
    s.insert("unassigned".into());
    s
}

fn template_sentence(c: &ClusterEvidence) -> String {
    let words = if c.incidence_words.is_empty() {
        String::new()
    } else {
        format!(
            " Keywords: {}.",
            c.incidence_words
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    if c.best_significant && c.coarse_label != "unassigned" {
        format!("{} is annotated as {}.{}", c.id, c.coarse_label, words)
    } else if c.best_label != "unassigned" {
        let q = c
            .best_q
            .map(|q| format!(" (best q={q:.3})"))
            .unwrap_or_default();
        format!(
            "{} has no significant call; closest panel type is {}{}.{}",
            c.id, c.best_label, q, words
        )
    } else {
        format!("{} is unassigned.{}", c.id, words)
    }
}

fn citation_check(draft: &str, allowed: &BTreeSet<String>, c: &ClusterEvidence) -> Result<String> {
    let lower = draft.to_lowercase();
    // Every evidence label for this cluster that appears in the sentence is fine;
    // reject if the sentence claims a type but names none of the allowed entities
    // for this cluster (coarse/best/incidence).
    let cluster_allowed: BTreeSet<String> = {
        let mut s = BTreeSet::new();
        if c.coarse_label != "unassigned" {
            s.insert(c.coarse_label.clone());
        }
        if c.best_label != "unassigned" {
            s.insert(c.best_label.clone());
        }
        s.extend(c.incidence_words.iter().cloned());
        s
    };
    if !cluster_allowed.is_empty() {
        let mentions = cluster_allowed
            .iter()
            .any(|e| lower.contains(&e.to_lowercase()));
        if !mentions {
            bail!(
                "citation check failed for {}: sentence names none of {:?}",
                c.id,
                cluster_allowed
            );
        }
    }
    // Reject tokens that look like other panel types from the run but aren't this cluster's.
    for ent in allowed {
        if ent == "unassigned" || cluster_allowed.contains(ent) {
            continue;
        }
        if lower.contains(&ent.to_lowercase())
            && !cluster_allowed.iter().any(|a| {
                a.to_lowercase().contains(&ent.to_lowercase())
                    || ent.to_lowercase().contains(&a.to_lowercase())
            })
        {
            // Only fail if this entity appears as a whole word-ish claim and isn't a substring
            // of an allowed label (e.g. "T" inside "Tcell").
            if ent.len() >= 3 {
                bail!(
                    "citation check failed for {}: sentence mentions `{}` outside evidence",
                    c.id,
                    ent
                );
            }
        }
    }
    if draft.contains('{') || draft.contains('}') {
        bail!("citation check failed for {}: braces in sentence", c.id);
    }
    Ok(draft.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn citation_accepts_template_with_best_label() {
        let c = ClusterEvidence {
            id: "K0".into(),
            coarse_label: "unassigned".into(),
            best_label: "NK".into(),
            best_q: Some(0.2),
            best_significant: false,
            incidence_words: vec!["killer".into()],
        };
        let allowed = allowed_entities(std::slice::from_ref(&c));
        let s = template_sentence(&c);
        assert!(citation_check(&s, &allowed, &c).is_ok());
        assert!(s.contains("NK"));
        assert!(s.contains("no significant"));
    }

    #[test]
    fn citation_rejects_foreign_panel_type() {
        let c = ClusterEvidence {
            id: "K0".into(),
            coarse_label: "Tcell".into(),
            best_label: "Tcell".into(),
            best_q: Some(0.01),
            best_significant: true,
            incidence_words: vec![],
        };
        let mut allowed = allowed_entities(std::slice::from_ref(&c));
        allowed.insert("Bcell".into());
        let bad = "K0 is annotated as Bcell.";
        assert!(citation_check(bad, &allowed, &c).is_err());
    }
}
