# legume-genomic-types

Shared genomic types and parsers used by [legume-rs](https://github.com/causalpathlab/legume-rs) tools (`faba`, `senna`, `chickpea`, `cnv`, …):

- GFF/GTF parsing
- BED intervals
- SAM/BAM barcode and strand helpers
- Transcript / exon models

Rust imports remain `genomic_data::…` (`[lib] name = "genomic_data"`).

```toml
legume-genomic-types = "0.4"
```

Developed in the legume-rs monorepo under `genomic-data/`.
