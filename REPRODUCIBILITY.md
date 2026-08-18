# Reproducibility

PASS 0 | MISMATCH 0 | BLOCKED 15 | PENDING 138

|         | id    | section   | claim                                                     | manuscript | computed | source |
| ------- | ----- | --------- | --------------------------------------------------------- | ---------- | -------- | ------ |
| BLOCKED | R3.01 | Results 3 | L2G average precision on held-out test set                | 0.81       |          |        |
| BLOCKED | R3.02 | Results 3 | L2G area under the curve                                  | 0.95       |          |        |
| BLOCKED | R3.03 | Results 3 | L2G recall at score >= 0.5                                | 0.65       |          |        |
| BLOCKED | R3.05 | Results 3 | previous Open Targets model average precision             | 0.65       |          |        |
| BLOCKED | R3.06 | Results 3 | L2G FDR at score >= 0.5 (%)                               | 11.5       |          |        |
| BLOCKED | R3.07 | Results 3 | nearest-gene FDR (%)                                      | 27         |          |        |
| BLOCKED | R3.12 | Results 3 | regions where a secondary CS carries the evidence (%)     | 81.5       |          |        |
| BLOCKED | R3.13 | Results 3 | eQTL colocalisation sensitivity (%)                       | 21.8       |          |        |
| BLOCKED | R3.14 | Results 3 | eQTL colocalisation FDR (%)                               | 65.1       |          |        |
| BLOCKED | R3.19 | Results 3 | OR for most LoF-constrained gene in CS neighbourhood      | 1.9        |          |        |
| BLOCKED | R5.20 | Results 5 | gPS univariate beta PAV association                       | 0.53       |          |        |
| BLOCKED | R5.21 | Results 5 | gPS univariate beta tissue specificity                    | -0.11      |          |        |
| BLOCKED | R5.23 | Results 5 | significantly enriched pathways (FDR < 0.05)              | 312        |          |        |
| BLOCKED | R5.24 | Results 5 | pathways enriched in high-pleiotropy genes                | 221        |          |        |
| BLOCKED | R5.25 | Results 5 | pathways enriched in low-pleiotropy genes                 | 91         |          |        |
| PENDING | R1.01 | Results 1 | GWAS studies analysed                                     | 100526     |          |        |
| PENDING | R1.02 | Results 1 | publications                                              | 4250       |          |        |
| PENDING | R1.03 | Results 1 | trait ontology terms                                      | 9280       |          |        |
| PENDING | R1.04 | Results 1 | therapeutic areas                                         | 23         |          |        |
| PENDING | R1.05 | Results 1 | pre-2017 studies >10% non-European (%)                    | 23.2       |          |        |
| PENDING | R1.06 | Results 1 | by end-2024 studies >10% non-European (%)                 | 35.1       |          |        |
| PENDING | R1.07 | Results 1 | GWAS credible sets                                        | 789453     |          |        |
| PENDING | R1.08 | Results 1 | studies with at least one CS                              | 39282      |          |        |
| PENDING | R1.09 | Results 1 | qualified CSs                                             | 520975     |          |        |
| PENDING | R1.10 | Results 1 | qualified disease CSs                                     | 70618      |          |        |
| PENDING | R1.11 | Results 1 | qualified measurement CSs                                 | 450357     |          |        |
| PENDING | R1.12 | Results 1 | unique variants in CSs                                    | 2024916    |          |        |
| PENDING | R1.13 | Results 1 | lead variants                                             | 211597     |          |        |
| PENDING | R1.14 | Results 1 | lead variants with PIP >= 0.9                             | 49772      |          |        |
| PENDING | R1.15 | Results 1 | molQTL CSs used for colocalisation                        | 2044305    |          |        |
| PENDING | R1.16 | Results 1 | tissues or cell types                                     | 98         |          |        |
| PENDING | R1.17 | Results 1 | CS-gene prioritisations                                   | 523409     |          |        |
| PENDING | R1.18 | Results 1 | gene-disease pairs                                        | 36858      |          |        |
| PENDING | R1.19 | Results 1 | gene-measurement pairs                                    | 150360     |          |        |
| PENDING | R1.20 | Results 1 | unique genes implicated                                   | 15641      |          |        |
| PENDING | R1.21 | Results 1 | disease-associated genes                                  | 8285       |          |        |
| PENDING | R1.22 | Results 1 | measurement-associated genes                              | 15160      |          |        |
| PENDING | R1.23 | Results 1 | diseases covered                                          | 1394       |          |        |
| PENDING | R1.24 | Results 1 | measurements covered                                      | 3412       |          |        |
| PENDING | R1.25 | Results 1 | protein-coding genes associated (%)                       | 77.9       |          |        |
| PENDING | R1.26 | Results 1 | genes not from EUR common studies by 2024                 | 2462       |          |        |
| PENDING | R1.27 | Results 1 | disease-associated genes by 2024 (denominator)            | 8129       |          |        |
| PENDING | R1.28 | Results 1 | genes from non-EUR studies                                | 1829       |          |        |
| PENDING | R1.29 | Results 1 | genes from mixed studies                                  | 633        |          |        |
| PENDING | R1.30 | Results 1 | gene-disease pairs not from EUR common                    | 16384      |          |        |
| PENDING | R1.31 | Results 1 | gene-disease pairs by 2024 (denominator)                  | 34905      |          |        |
| PENDING | R1.32 | Results 1 | gene-disease pairs from non-EUR                           | 12285      |          |        |
| PENDING | R1.33 | Results 1 | gene-disease pairs from mixed                             | 4099       |          |        |
| PENDING | R1.34 | Results 1 | IRR non-EUR vs EUR                                        | 1.69       |          |        |
| PENDING | R1.35 | Results 1 | IRR mixed vs EUR                                          | 0.97       |          |        |
| PENDING | R1.36 | Results 1 | IRR per tenfold effective sample size                     | 3.67       |          |        |
| PENDING | R1.37 | Results 1 | genes associated with 2+ diseases                         | 5314       |          |        |
| PENDING | R1.38 | Results 1 | genes associated with >1 therapeutic area                 | 4743       |          |        |
| PENDING | R2.01 | Results 2 | non-redundant replicated CSs with PIP >= 0.5              | 120809     |          |        |
| PENDING | R2.02 | Results 2 | enhancer-overlapping share in GWAS disease loci (%)       | 13.8       |          |        |
| PENDING | R2.03 | Results 2 | promoter-overlapping share in cis-pQTLs (%)               | 9.1        |          |        |
| PENDING | R2.04 | Results 2 | promoter-overlapping share in GWAS disease loci (%)       | 2.9        |          |        |
| PENDING | R2.05 | Results 2 | replicated lead variants intragenic or PAV (%)            | 63         |          |        |
| PENDING | R3.04 | Results 3 | CS-protein coding gene pairs scored                       | 7066749    |          |        |
| PENDING | R3.08 | Results 3 | prioritised genes lacking PAV/molQTL support in 2015 (%)  | 49         |          |        |
| PENDING | R3.09 | Results 3 | prioritised genes lacking PAV/molQTL support in 2024 (%)  | 26         |          |        |
| PENDING | R3.10 | Results 3 | prioritised genes with eQTL colocalisation (%)            | 36.7       |          |        |
| PENDING | R3.11 | Results 3 | prioritised genes with pQTL colocalisation (%)            | 5.7        |          |        |
| PENDING | R3.15 | Results 3 | CS-gene assignments that are nearest to TSS (%)           | 81.2       |          |        |
| PENDING | R3.16 | Results 3 | nearest-gene assignments with no PAV or coloc support (%) | 46.1       |          |        |
| PENDING | R3.17 | Results 3 | assignments supported by a PAV (%)                        | 13.0       |          |        |
| PENDING | R3.18 | Results 3 | MST1 L2G score                                            | 0.86       |          |        |
| PENDING | R4.01 | Results 4 | independent colocalisation clusters                       | 20041      |          |        |
| PENDING | R4.02 | Results 4 | clusters with more than one lead variant                  | 5595       |          |        |
| PENDING | R4.03 | Results 4 | clusters linked to multiple diseases                      | 6617       |          |        |
| PENDING | R4.04 | Results 4 | maximum diseases per cluster                              | 120        |          |        |
| PENDING | R4.05 | Results 4 | mean diseases per cluster                                 | 2.14       |          |        |
| PENDING | R4.06 | Results 4 | clusters linked to multiple therapeutic areas             | 4539       |          |        |
| PENDING | R4.07 | Results 4 | maximum therapeutic areas per cluster                     | 20         |          |        |
| PENDING | R4.08 | Results 4 | mean therapeutic areas per cluster                        | 1.40       |          |        |
| PENDING | R4.09 | Results 4 | Spearman rho diseases vs therapeutic areas per cluster    | 0.81       |          |        |
| PENDING | R4.10 | Results 4 | vPS variance explained by predicted power alone (%)       | 14.7       |          |        |
| PENDING | R4.11 | Results 4 | vPS variance explained by full joint model (%)            | 17.7       |          |        |
| PENDING | R4.12 | Results 4 | vPS variance explained excluding predicted power (%)      | 6.0        |          |        |
| PENDING | R4.13 | Results 4 | vPS variance explained by max effective sample size (%)   | 0.45       |          |        |
| PENDING | R4.14 | Results 4 | pleiotropic lead variants                                 | 5188       |          |        |
| PENDING | R4.15 | Results 4 | lead variants fully concordant in direction               | 4797       |          |        |
| PENDING | R4.16 | Results 4 | lead variants with an opposing direction                  | 391        |          |        |
| PENDING | R4.17 | Results 4 | lead variants with lead vPS >= 10                         | 135        |          |        |
| PENDING | R4.18 | Results 4 | of those, with directionality agreement < 0.8             | 31         |          |        |
| PENDING | R4.19 | Results 4 | genes carrying those 31 variants                          | 34         |          |        |
| PENDING | R4.20 | Results 4 | APOE 19_44908684_T_C lead vPS                             | 85         |          |        |
| PENDING | R4.21 | Results 4 | APOE 19_44908684_T_C beta concordance                     | 0.66       |          |        |
| PENDING | R4.22 | Results 4 | APOE 19_44908684_T_C therapeutic areas                    | 15         |          |        |
| PENDING | R5.01 | Results 5 | disease genes with gPS > 1                                | 5314       |          |        |
| PENDING | R5.02 | Results 5 | mean gPS                                                  | 4.45       |          |        |
| PENDING | R5.03 | Results 5 | maximum gPS                                               | 148        |          |        |
| PENDING | R5.04 | Results 5 | genes linked to multiple therapeutic areas                | 4743       |          |        |
| PENDING | R5.05 | Results 5 | mean therapeutic areas per gene                           | 2.53       |          |        |
| PENDING | R5.06 | Results 5 | maximum therapeutic areas per gene                        | 21         |          |        |
| PENDING | R5.07 | Results 5 | Spearman rho gPS vs therapeutic area count                | 0.92       |          |        |
| PENDING | R5.08 | Results 5 | FTO gPS                                                   | 126        |          |        |
| PENDING | R5.09 | Results 5 | APOE gPS                                                  | 107        |          |        |
| PENDING | R5.10 | Results 5 | ABO gPS                                                   | 105        |          |        |
| PENDING | R5.11 | Results 5 | CDKN2B gPS                                                | 148        |          |        |
| PENDING | R5.12 | Results 5 | CDKN2B therapeutic areas                                  | 21         |          |        |
| PENDING | R5.13 | Results 5 | Spearman rho variants per gene vs gPS                     | 0.87       |          |        |
| PENDING | R5.14 | Results 5 | gPS joint model Pearson R2                                | 0.15       |          |        |
| PENDING | R5.15 | Results 5 | gPS univariate beta max sample size                       | 1.82       |          |        |
| PENDING | R5.16 | Results 5 | gPS univariate beta LoF constraint                        | 0.64       |          |        |
| PENDING | R5.17 | Results 5 | gPS univariate beta missense constraint                   | 0.59       |          |        |
| PENDING | R5.18 | Results 5 | gPS univariate beta pathway count                         | 3.45       |          |        |
| PENDING | R5.19 | Results 5 | gPS univariate beta gene length                           | 2.17       |          |        |
| PENDING | R5.22 | Results 5 | Pearson r LoF vs missense constraint                      | 0.628      |          |        |
| PENDING | R5.26 | Results 5 | gene sets tested                                          | 21         |          |        |
| PENDING | R5.27 | Results 5 | gene sets significantly associated with higher gPS        | 10         |          |        |
| PENDING | R5.28 | Results 5 | OR cancer driver genes per doubling of gPS                | 1.49       |          |        |
| PENDING | R5.29 | Results 5 | OR mouse knockout-lethal homologs                         | 1.30       |          |        |
| PENDING | R5.30 | Results 5 | OR developmental disorder panel genes                     | 1.32       |          |        |
| PENDING | R5.31 | Results 5 | OR safety-terminated trial targets                        | 1.31       |          |        |
| PENDING | R5.32 | Results 5 | OR withdrawn drug targets                                 | 1.08       |          |        |
| PENDING | R5.33 | Results 5 | withdrawn drug target gene set size                       | 166        |          |        |
| PENDING | R5.34 | Results 5 | OR known safety event targets                             | 1.13       |          |        |
| PENDING | R5.35 | Results 5 | known safety event gene set size                          | 214        |          |        |
| PENDING | R5.36 | Results 5 | OR LoF-constrained genes Q4                               | 1.23       |          |        |
| PENDING | R5.37 | Results 5 | OR low-constraint genes Q1                                | 0.85       |          |        |
| PENDING | R5.38 | Results 5 | OR human knockout genes                                   | 0.89       |          |        |
| PENDING | R6.01 | Results 6 | approved target-indication pairs with GWAS support        | 242        |          |        |
| PENDING | R6.02 | Results 6 | overall genetic support OR                                | 3.62       |          |        |
| PENDING | R6.03 | Results 6 | overall relative success                                  | 2.76       |          |        |
| PENDING | R6.04 | Results 6 | OR after mixed-effects adjustment                         | 3.14       |          |        |
| PENDING | R6.05 | Results 6 | OR rare-variant associations                              | 7.0        |          |        |
| PENDING | R6.06 | Results 6 | OR common-variant associations                            | 3.4        |          |        |
| PENDING | R6.07 | Results 6 | P rare vs common                                          | 0.0077     |          |        |
| PENDING | R6.08 | Results 6 | OR Orphanet                                               | 5.1        |          |        |
| PENDING | R6.09 | Results 6 | OR OMIM                                                   | 4.7        |          |        |
| PENDING | R6.10 | Results 6 | OR ClinVar/ClinGen                                        | 5.1        |          |        |
| PENDING | R6.11 | Results 6 | OR UniProt                                                | 5.0        |          |        |
| PENDING | R6.12 | Results 6 | OR Genomics England PanelApp                              | 4.5        |          |        |
| PENDING | R6.13 | Results 6 | OR PAV-supported associations                             | 6.0        |          |        |
| PENDING | R6.14 | Results 6 | OR non-PAV associations                                   | 3.1        |          |        |
| PENDING | R6.15 | Results 6 | P PAV vs non-PAV                                          | 0.0002     |          |        |
| PENDING | R6.16 | Results 6 | OR gene-based analyses                                    | 7.2        |          |        |
| PENDING | R6.17 | Results 6 | OR large effect                                           | 4.6        |          |        |
| PENDING | R6.18 | Results 6 | OR small effect                                           | 3.5        |          |        |
| PENDING | R6.19 | Results 6 | OR gPS <= 5                                               | 4.8        |          |        |
| PENDING | R6.20 | Results 6 | OR gPS >= 10                                              | 3.0        |          |        |
| PENDING | R6.21 | Results 6 | P gPS low vs high                                         | 0.008      |          |        |
| PENDING | R6.22 | Results 6 | OR single therapeutic area                                | 4.3        |          |        |
| PENDING | R6.23 | Results 6 | OR six or more therapeutic areas                          | 2.9        |          |        |
| PENDING | R6.24 | Results 6 | P TA count low vs high                                    | 0.18       |          |        |
| PENDING | R6.25 | Results 6 | OR high pleiotropy vs no GWAS support                     | 0.74       |          |        |
| PENDING | R6.26 | Results 6 | OR previously approved target                             | 4.13       |          |        |
| PENDING | R6.27 | Results 6 | gene-disease associations meeting PAV + 2-5 TA definition | 2734       |          |        |
| PENDING | R6.28 | Results 6 | OR PAV + 2-5 therapeutic areas                            | 10.3       |          |        |
| PENDING | R6.29 | Results 6 | RS PAV + 2-5 therapeutic areas                            | 4.8        |          |        |
| PENDING | R6.30 | Results 6 | approved pairs also meeting the strict definition         | 51         |          |        |
| PENDING | R6.31 | Results 6 | that subset as a share of 242 (%)                         | 21.1       |          |        |
