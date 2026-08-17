//! `senna docs` — the method write-ups, compiled into the binary.
//!
//! `include_str!`, not paths read at runtime. The binary is often the only thing on the machine
//! that ran the analysis (installed with `cargo install`, or copied to a cluster with no checkout
//! beside it), and a doc you cannot reach from there is a doc nobody reads. It also means the
//! build breaks if one of these files is moved or deleted — which enforces that they *exist*,
//! though not that they are *current*.

use anyhow::Result;
use clap::builder::PossibleValue;
use clap::{Args, ValueEnum};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Topic {
    /// Marker cell-type annotation, end to end.
    Annotation,
    /// Why the annotation pools cells into coarse clusters.
    Grouping,
    /// Reference-based deconvolution of pseudobulk profiles.
    Deconvolve,
    /// PLAN, not implemented — annotation onto a Cell Ontology DAG.
    OntologyPlan,
    /// PLAN, not implemented — expert knowledge for lineage rooting.
    RootingPlan,
}

/// Every write-up, in one place: the topic, a one-line blurb, and the text.
///
/// The listing `senna docs` prints and the text `senna docs <TOPIC>` prints are both read from
/// here, so the index can never advertise a topic the command cannot serve — which is exactly
/// what happens when the two are maintained separately.
const DOCS: &[(Topic, &str, &str)] = &[
    (
        Topic::Annotation,
        "METHOD  marker cell-type annotation, end to end",
        include_str!("../docs/annotation-methods.md"),
    ),
    (
        Topic::Grouping,
        "METHOD  why the annotation pools cells into coarse clusters",
        include_str!("../docs/annotation-grouping.md"),
    ),
    (
        Topic::Deconvolve,
        "METHOD  reference-based deconvolution of pseudobulk profiles",
        include_str!("../docs/deconvolve.md"),
    ),
    (
        Topic::OntologyPlan,
        "PLAN    (not implemented) annotation onto a Cell Ontology DAG",
        include_str!("../docs/annotation-ontology-plan.md"),
    ),
    (
        Topic::RootingPlan,
        "PLAN    (not implemented) expert knowledge for lineage rooting",
        include_str!("../docs/lineage-rooting.md"),
    ),
];

#[derive(Args, Debug)]
pub struct DocsArgs {
    #[arg(
        value_enum,
        help = "Which write-up to print (omit to list what there is)"
    )]
    pub topic: Option<Topic>,
}

pub fn run_docs(args: &DocsArgs) -> Result<()> {
    let Some(want) = args.topic else {
        println!("senna method write-ups (`senna docs <TOPIC>` to read one):\n");
        for (topic, blurb, _) in DOCS {
            // The slug clap will actually ACCEPT. Deriving it with `format!("{topic:?}")` prints
            // `ontologyplan` while the parser wants `ontology-plan`, so the listing would
            // advertise names the command then refuses — the one failure this table prevents.
            let slug = topic
                .to_possible_value()
                .as_ref()
                .map(PossibleValue::get_name)
                .unwrap_or_default()
                .to_string();
            println!("  {slug:<14} {blurb}");
        }
        println!(
            "\nThe per-cell feature matrices these commands read are built by `faba`; \
             see `faba docs profiling`.\n"
        );
        return Ok(());
    };
    let text = DOCS
        .iter()
        .find(|(t, _, _)| *t == want)
        .map(|(_, _, text)| *text)
        .expect("every Topic variant has a row in DOCS");
    println!("{text}");
    Ok(())
}
