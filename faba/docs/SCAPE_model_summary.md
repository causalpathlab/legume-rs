# SCAPE Model Summary

## Publication Information

**Title:** SCAPE: a mixture model revealing single-cell polyadenylation diversity and cellular dynamics during cell differentiation and reprogramming

**Authors:** Ran Zhou, Xia Xiao, Ping He, Yuancun Zhao, Mengying Xu, Xiuran Zheng, Ruirui Yang, Shasha Chen, Lifang Zhou, Dan Zhang, Qingxin Yang, Junwei Song, Chao Tang, Yiming Zhang, Jing-wen Lin, Lu Cheng, Lu Chen (corresponding author)

**Journal:** Nucleic Acids Research, 2022, Vol. 50, No. 11, Page e66

**DOI:** https://doi.org/10.1093/nar/gkac167

**Published:** March 14, 2022

---

## Overview

SCAPE is a Bayesian probabilistic mixture model designed to identify and quantify alternative polyadenylation (APA) sites at single-cell resolution from scRNA-seq data. It addresses the challenge of detecting transcript diversity at the 3' end, which plays important regulatory roles in cell differentiation, embryonic development, and cancer progression.

---

## Problem Statement

SCAPE solves three key limitations in existing APA analysis methods:

1. **Inaccurate pA site location inference** - Most scRNA-seq reads don't directly cover cleavage sites
2. **Difficulty separating overlapping peaks** - Multiple pA sites in close proximity are hard to distinguish
3. **Weak signal detection** - Challenging to differentiate true biological signals from technical noise

---

## Model Architecture

### Statistical Framework

SCAPE employs a **Bayesian probabilistic mixture model** that treats scRNA-seq reads as originating from:
- **K isoform components** - Each representing a distinct polyadenylation site
- **One noise component** - Accounting for random, non-informative reads

### Key Innovation

The model leverages **insert size information** from paired-end sequencing libraries to infer pA positions, rather than relying solely on read coverage at cleavage sites.

---

## Mixture Model Components

### Mathematical Formulation

The probability of observing a read with insert size $x$ is modeled as a mixture:

$$P(x) = \sum_{k=1}^{K} \pi_k \cdot f_k(x | \mu_k, \sigma_k) + \pi_0 \cdot f_0(x)$$

where:
- $K$ is the number of APA isoform components
- $f_k(x | \mu_k, \sigma_k)$ is the probability density for isoform $k$
- $f_0(x)$ is the uniform noise distribution
- $\sum_{k=0}^{K} \pi_k = 1$ (mixture weights sum to 1)

### APA Isoform Components (K components)

Each isoform component $k$ models reads arising from a specific polyadenylation site. The component is characterized by three parameters:

- $\mu_k$ - Mean position of the polyadenylation site
- $\sigma_k$ - Standard deviation capturing pA site positional fluctuation
- $\pi_k$ - Component weight/proportion (relative abundance)

The probability density $f_k(x | \mu_k, \sigma_k)$ incorporates:
- Gaussian fluctuation of pA site position around $\mu_k$
- Distribution of poly(A) tail lengths
- Fragment size distribution from the sequencing protocol

### Noise Component

- Uniformly distributed across the 3' UTR region
- Captures random, non-informative sequencing reads
- Helps distinguish true biological signals from technical artifacts

---

## Model Parameters and Assumptions

### Core Assumptions

1. **Fragment size distribution:**

   cDNA fragment sizes follow a Gaussian distribution:

   $$L_{\text{cDNA}} \sim \mathcal{N}(\mu_{\text{frag}}, \sigma_{\text{frag}}^2)$$

   Typical values:
   - $\mu_{\text{frag}} \approx 300$ bp
   - $\sigma_{\text{frag}} \approx 50$ bp

2. **pA site variability:**

   Polyadenylation sites exhibit Gaussian fluctuation around their mean position:

   $$p_k \sim \mathcal{N}(\mu_k, \sigma_k^2)$$

   where $\sigma_k$ captures the inherent variability in cleavage site selection for isoform $k$

3. **poly(A) tail lengths:**

   Poly(A) tail lengths follow empirical distributions:

   $$L_{\text{poly(A)}} \sim f_{\text{empirical}}$$

   - Typically range from 20-150 nucleotides
   - Can be estimated from data or literature values
   - Distribution varies by cell type and experimental protocol

4. **Fragment length calculation:**

   The observed insert size is the sum of the 3' UTR distance and poly(A) tail:

   $$L_{\text{insert}} = L_{\text{3' UTR}} + L_{\text{poly(A)}}$$

### Hidden Variables

The model marginalizes over three sources of uncertainty:

1. cDNA fragment size variation
2. pA site positional fluctuation
3. poly(A) tail length heterogeneity

---

## Inference Algorithm

### Expectation-Maximization (EM)

SCAPE uses an **approximate EM algorithm** for parameter inference:

- **E-step:** Calculate the posterior probability (responsibility) that read $i$ belongs to component $k$:

  $$\gamma_{ik} = \frac{\pi_k \cdot f_k(x_i | \mu_k, \sigma_k)}{\sum_{j=0}^{K} \pi_j \cdot f_j(x_i | \mu_j, \sigma_j)}$$

- **M-step:** Update parameters to maximize expected log-likelihood:
  - $\pi_k$ has closed-form update: $\pi_k = \frac{1}{N} \sum_{i=1}^{N} \gamma_{ik}$
  - $\mu_k$ and $\sigma_k$ require numerical optimization (no analytical solutions exist)

### Model Selection

**Bayesian Information Criterion (BIC)** is used to automatically determine the optimal number of isoform components:

$$\text{BIC} = -2 \ln(L) + p \ln(n)$$

where:
- $L$ is the maximized likelihood of the model
- $p$ is the number of parameters
- $n$ is the number of observations (reads)

The model with the lowest BIC is selected, balancing model fit with complexity penalty.

---

## Input Data Requirements

### Required Inputs

1. **Genomic alignment data:**
   - Genome-aligned BAM files
   - From poly(A)-enriched scRNA-seq protocols
   - Compatible platforms: 10× Genomics, Microwell-seq, Drop-seq

2. **Fragment length parameters:**
   - Mean fragment length (from experimental protocol)
   - Standard deviation of fragment length

3. **Model configuration:**
   - Maximum number of potential isoforms per gene
   - Minimum weight threshold for isoform components (filters low-abundance isoforms)

### Optional Inputs

- Poly(A) length distribution (can be estimated from data or literature)

---

## Model Outputs

### Per-Gene Parameters

1. **pA site locations ($\mu_k$)** - Mean position of each identified polyadenylation site
2. **Position confidence intervals ($\sigma_k$)** - Uncertainty in pA site locations
3. **Isoform proportions ($\pi_k$)** - Relative abundance of each APA isoform
4. **Read assignments** - Component membership for individual reads

### Derived Metrics

1. **Expected pA length:**

   Normalized index summarizing the overall APA pattern per cell/gene:

   $$\text{Expected pA length} = \frac{\sum_{k=1}^{K} \pi_k \cdot \mu_k - \mu_{\min}}{\mu_{\max} - \mu_{\min}}$$

   - Range: 0 (proximal usage) to 1 (distal usage)
   - Weighted average of pA site positions normalized to [0,1]

2. **pA counts per site:**
   - Enables differential APA analysis
   - Can be used for statistical comparisons across conditions

3. **APA isoform classifications:**
   - L-shaped (proximal preference)
   - J-shaped (distal preference)
   - Overdispersed
   - Underdispersed
   - Multimodal

### Functional Outputs

1. **Differential APA analysis** - Identify changes in APA usage between conditions
2. **Cell clustering improvements** - Enhanced cell type separation using APA features
3. **Trajectory inference** - Incorporate APA dynamics into developmental trajectories

---

## Performance Validation

### Discovery Statistics

From analysis of 36 mouse organs:
- **31,558 pA sites identified**
- **43.8% (13,807) were novel** (not in existing annotations)

### Validation Results

- **93.9% of unannotated sites** validated by long-read sequencing
- **Superior performance** compared to competing methods
- **Higher F-scores** across precision-recall curves

---

## Implementation Notes

### Compatible with Standard Workflows

- Works with existing scRNA-seq preprocessing pipelines
- Takes standard BAM files as input
- Integrates with downstream analysis tools

### Computational Approach

- Bayesian framework provides principled uncertainty quantification
- EM algorithm enables scalable inference
- Automatic model selection reduces manual parameter tuning

---

## Applications

1. **Cell differentiation studies** - Track APA changes during lineage commitment
2. **Reprogramming research** - Monitor APA dynamics in iPSC generation
3. **Cancer biology** - Identify APA alterations in tumor progression
4. **Developmental biology** - Characterize APA regulation in embryogenesis

---

## References

Zhou, R., Xiao, X., He, P., Zhao, Y., Xu, M., Zheng, X., Yang, R., Chen, S., Zhou, L., Zhang, D., Yang, Q., Song, J., Tang, C., Zhang, Y., Lin, J.W., Cheng, L., & Chen, L. (2022). SCAPE: a mixture model revealing single-cell polyadenylation diversity and cellular dynamics during cell differentiation and reprogramming. *Nucleic Acids Research*, 50(11), e66. https://doi.org/10.1093/nar/gkac167

**Available at:**
- Oxford Academic: https://academic.oup.com/nar/article/50/11/e66/6548409
- PubMed Central: https://pmc.ncbi.nlm.nih.gov/articles/PMC9226526/
- PubMed: https://pubmed.ncbi.nlm.nih.gov/35288753/
