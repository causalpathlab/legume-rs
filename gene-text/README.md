# gene-text

The text side of the feature graph. A small BERT-family encoder from the
Hugging Face Hub reads descriptions of genes, ontology terms and pathways;
a document-frequency vocabulary keeps the words worth a node; and every
feature is linked to the words of its own text, weighted by the encoder's
contextual similarity times TF-IDF. The edges feed `senna fne --edges`; the
pooled vectors are a text prior for the gene side.

```
lupin text-qc --uniprot-tsv human.tsv --obo go-basic.obo -o run          # inspect the word cuts (optional)
lupin word-graph --uniprot-tsv human.tsv --obo go-basic.obo -o run       # encode, write the text graph
senna fne biogrid.tsv --edges run.feature_word.edges.tsv,run.knn_graph.edges.tsv --relation-weight gene:word=0.5 -o graph
```

## Inputs

| Flag | Source | Where to get it |
|---|---|---|
| `--uniprot-tsv` | reviewed human proteins: symbol, protein name, function | `https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true&fields=gene_primary,gene_synonym,protein_name,cc_function&format=tsv` |
| `--gene-info` | NCBI symbol, synonyms, full name (no summary) | `https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz` |
| `--obo` | ontology term names and definitions | `http://purl.obolibrary.org/obo/go/go-basic.obo`, `http://purl.obolibrary.org/obo/cl/cl-basic.obo` |
| `--gmt` | gene-set names and descriptions (MSigDB; a URL description is ignored) | `https://www.gsea-msigdb.org/gsea/msigdb/human/download_geneset.jsp` |
| `--text` | `feature <TAB> type <TAB> name <TAB> text` | what `senna fne --export-text` writes |

Bare symbols carry almost no signal for a text encoder; give it function
text (UniProt) or definitions (OBO) whenever possible.

## Model

`--model` takes any BERT-family encoder id (default
`sentence-transformers/all-MiniLM-L6-v2`; `BAAI/bge-small-en-v1.5` is a
stronger small choice). Files are fetched once into the Hugging Face cache;
`--model-dir` points at a local directory instead. `--device cuda` needs a
`--features cuda` build.

## Vocabulary QC

Words are lowercased, filtered by length and stopword lists (NLTK English,
a built-in biomedical filler list, `--extra-stopwords`), optionally stemmed,
then both tails of the document-frequency distribution are cut by quantile
(`--df-lower-quantile`, `--df-upper-quantile`) with absolute limits on top
(`--min-df`, `--max-df-frac`). The run prints the df histogram and the words
on each side of every cut; `{out}.vocab.tsv` lists every candidate with its
verdict and can be edited and handed back through `knn-graph --vocab-file`.
`lupin word-graph` runs the same step itself; `lupin text-qc` only lets you look first.

## Outputs

- `{out}.feature_word.edges.tsv` — `type feature word <word> weight`, top
  `--words-per-feature` per feature; weight = max(cos, 0) · (1 + ln tf) · idf.
- `{out}.feature_word_expanded.edges.tsv` (`--expand-k`) — nearest vocabulary
  words a feature's text lacks, by CSLS.
- `{out}.knn_graph.edges.tsv` — feature–feature text similarity, `--knn` per feature (default 10).
- `{out}.text_embedding.parquet` — pooled vectors, centred; `{out}.feature_types.parquet`.
- `{out}.vocab.tsv`, `{out}.feature_text.tsv` — the vocabulary and the corpus as read.
