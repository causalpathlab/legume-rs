# lupin

**L**abel, **U**nfold, **P**lace, **I**nterpret, **N**arrate

Text graphs, cell-type annotation, lineage, and short citation-checked descriptions.

```sh
lupin text-qc --uniprot-tsv human.tsv --obo go-basic.obo -o run
lupin word-graph --uniprot-tsv human.tsv --obo go-basic.obo -o run   # alias: vocab-graph
lupin annotate -f run.senna.json -m markers.tsv -o out              # --method auto|enrichment|projection
lupin lineage -f out/gem -o out/lin
lupin lineage-plot -f out/lin
lupin dyn-assoc -f out/lin -s sites.zarr.zip --modality m6a -o out/assoc
lupin pseudotime -f run.senna.json -o out
lupin describe -f out --text-prefix run -o out
```

Former homes: `gene-text` → `text-qc` / `word-graph`; `senna`/`pinto` annotate and lineage family → stubs pointing here.
