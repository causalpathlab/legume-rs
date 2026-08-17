//! `faba docs` — the method write-ups, compiled into the binary.
//!
//! `include_str!`, not paths read at runtime. The binary is often the only thing on the machine
//! that ran the analysis (installed with `cargo install`, or copied to a cluster with no checkout
//! beside it), and a doc you cannot reach from there is a doc nobody reads. It also means the
//! build breaks if the file is moved or deleted — which enforces that it *exists*, though not
//! that it is *current*.
//!
//! The annotation and lineage write-ups live in `senna docs`, with the subcommands they describe.

use anyhow::Result;
use clap::builder::PossibleValue;
use clap::{Args, ValueEnum};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Topic {
    /// BAM to per-cell features: m6A, A-to-I, APA, gene counts, SNPs.
    Profiling,
}

/// Every write-up, in one place: the topic, a one-line blurb, and the text.
///
/// The listing `faba docs` prints and the text `faba docs <TOPIC>` prints are both read from
/// here, so the index can never advertise a topic the command cannot serve — which is exactly
/// what happens when the two are maintained separately.
const DOCS: &[(Topic, &str, &str)] = &[(
    Topic::Profiling,
    "METHOD  BAM to per-cell features: m6A, A-to-I, APA, gene counts, SNPs",
    include_str!("../docs/profiling-methods.md"),
)];

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
        println!("faba method write-ups (`faba docs <TOPIC>` to read one):\n");
        for (topic, blurb, _) in DOCS {
            // The slug clap will actually ACCEPT, not the Debug spelling — see the same
            // loop in `senna docs`, where the two diverge and a derived slug once advertised
            // a name the parser then refused.
            let slug = topic
                .to_possible_value()
                .as_ref()
                .map(PossibleValue::get_name)
                .unwrap_or_default()
                .to_string();
            println!("  {slug:<14} {blurb}");
        }
        println!("\nAnnotation and lineage methods moved with their subcommands: `senna docs`.\n");
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
