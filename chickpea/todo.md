
Two types of regression analysis + simulation codes

# task 1. regressing each gene on the genotypes found in cis-regulatory regions

- we can take two data: (1) data backend for single cell data (2) genotype matrix, at first just consider plink bed, bim, fam format or plink2 format (don't try to parse yourself use existing crate.io)

- additionally we can take cell type annotations, or probabilistic annotation matrix which we can obtain from senna analysis

- we also need to take cell to individual assignment file so that we can map cells to individuals, while stratifying by cell types, into pseudo bulk data. to be matched with genotype data

- since we tried different types of regression approaches in the candle-util, we can start using them with sgd-based model fitting in parallel across all the genes (or a minbatch of genes, or chromsome-by-chromosome, and so on)

- of course we would want susie prior

- we would want a seprate simulation cli based on geontpyes
  - there, we can take heritability parameters
  - we can take number of causal variants

# task 2. probably similar, but want to mach peaks to genes in multiome data

- we can take two series (vectors) of data backend files 

- we can construct pseudobulk data on both sides

- then, we can fit regression models on the psuedobulk data 

- of course we would want simulation routine separate cli



