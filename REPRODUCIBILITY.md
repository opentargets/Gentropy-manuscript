# Reproducibility

PASS 602 | MISMATCH 55 | BLOCKED 9 | PRECOMPUTED 7 | PENDING 0

|             | id      | section   | claim                                                           | manuscript | computed               | source                          |
| ----------- | ------- | --------- | --------------------------------------------------------------- | ---------- | ---------------------- | ------------------------------- |
| MISMATCH    | R2.01   | Results 2 | non-redundant replicated CSs with PIP >= 0.5                    | 120809     | 121490                 | selective_pressures.json        |
| MISMATCH    | R3.09   | Results 3 | prioritised genes lacking PAV/molQTL support in 2024 (%)        | 26         | 40.0                   | colocalisation_l2g.json         |
| MISMATCH    | R3.12   | Results 3 | regions where a secondary CS carries the evidence (%)           | 81.5       | 50.7                   | sr05_secondary_signals.json     |
| MISMATCH    | R3.17   | Results 3 | assignments supported by a PAV (%)                              | 13.0       | 12.1                   | colocalisation_l2g.json         |
| MISMATCH    | R4.06   | Results 4 | clusters linked to multiple therapeutic areas                   | 4539       | 4766                   | variant_pleiotropy.json         |
| MISMATCH    | R4.07   | Results 4 | maximum therapeutic areas per cluster                           | 20         | 19                     | variant_pleiotropy.json         |
| MISMATCH    | R4.08   | Results 4 | mean therapeutic areas per cluster                              | 1.40       | 1.45                   | variant_pleiotropy.json         |
| MISMATCH    | R4.09   | Results 4 | Spearman rho diseases vs therapeutic areas per cluster          | 0.81       | 0.84                   | variant_pleiotropy.json         |
| MISMATCH    | R4.10   | Results 4 | vPS variance explained by predicted power alone (%)             | 14.7       | 15.073413831782467     | variant_pleiotropy.json         |
| MISMATCH    | R4.11   | Results 4 | vPS variance explained by full joint model (%)                  | 17.7       | 16.946450149990614     | variant_pleiotropy.json         |
| MISMATCH    | R4.12   | Results 4 | vPS variance explained excluding predicted power (%)            | 6.0        | 3.826953106634868      | variant_pleiotropy.json         |
| MISMATCH    | R4.13   | Results 4 | vPS variance explained by max effective sample size (%)         | 0.45       | 0.5182255329135724     | variant_pleiotropy.json         |
| MISMATCH    | R4.14   | Results 4 | pleiotropic lead variants                                       | 5188       | 2166                   | variant_pleiotropy.json         |
| MISMATCH    | R4.15   | Results 4 | lead variants fully concordant in direction                     | 4797       | 1844                   | variant_pleiotropy.json         |
| MISMATCH    | R4.16   | Results 4 | lead variants with an opposing direction                        | 391        | 322                    | variant_pleiotropy.json         |
| MISMATCH    | R4.17   | Results 4 | lead variants with lead vPS >= 10                               | 135        | 67                     | variant_pleiotropy.json         |
| MISMATCH    | R4.18   | Results 4 | of those, with directionality agreement < 0.8                   | 31         | 18                     | variant_pleiotropy.json         |
| MISMATCH    | R4.19   | Results 4 | genes carrying those 31 variants                                | 34         | 21                     | variant_pleiotropy.json         |
| MISMATCH    | R4.20   | Results 4 | APOE 19_44908684_T_C lead vPS                                   | 85         | 71                     | variant_pleiotropy.json         |
| MISMATCH    | R4.21   | Results 4 | APOE 19_44908684_T_C beta concordance                           | 0.66       | 0.56                   | variant_pleiotropy.json         |
| MISMATCH    | R5.16   | Results 5 | gPS univariate beta LoF constraint                              | 0.64       | 0.616682987515627      | gene_pleiotropy.json            |
| MISMATCH    | R5.17   | Results 5 | gPS univariate beta missense constraint                         | 0.59       | 0.8590247266577721     | gene_pleiotropy.json            |
| MISMATCH    | R6.08   | Results 6 | OR Orphanet                                                     | 5.1        | 5.0                    | therapeutic_success.json        |
| MISMATCH    | R6.09   | Results 6 | OR OMIM                                                         | 4.7        | 5.3                    | therapeutic_success.json        |
| MISMATCH    | R6.25   | Results 6 | OR high pleiotropy vs no GWAS support                           | 0.74       | 2.97                   | therapeutic_success.json        |
| MISMATCH    | R6.26   | Results 6 | OR previously approved target                                   | 4.13       | 8.71                   | therapeutic_success.json        |
| MISMATCH    | S1.16   | SR 1      | slope of credible set size on lead variant MAF                  | 14.0       | 25.969                 | sr01_finemapping_catalogue.json |
| MISMATCH    | S1.29   | SR 1      | lead variant-disease pairs seen in more than one study (%)      | 16         | 16.9                   | sr01_finemapping_catalogue.json |
| MISMATCH    | S1.30   | SR 1      | of those, significant Cochran heterogeneity (%)                 | 15         | 10.4                   | sr01_finemapping_catalogue.json |
| MISMATCH    | S11.106 | SR 11     | ceiling, ChEMBL PAV low pleiotropy OR                           | 10.32      | 10.59                  | sr11_external.json              |
| MISMATCH    | S11.107 | SR 11     | ceiling, ChEMBL PAV high pleiotropy OR                          | 3.1        | 3.18                   | sr11_external.json              |
| MISMATCH    | S11.114 | SR 11     | ceiling, Pharmaprojects PAV low OR                              | 4.5        | 4.58                   | sr11_external.json              |
| MISMATCH    | S11.115 | SR 11     | ceiling, Pharmaprojects PAV high OR                             | 1.09       | 1.11                   | sr11_external.json              |
| MISMATCH    | S3.18   | SR 3      | selectivity at L2G >= 0.5                                       | 0.994      | 0.995                  | sr03_l2g.json                   |
| MISMATCH    | S3.19   | SR 3      | recall at L2G >= 0.5                                            | 0.645      | 0.646                  | sr03_l2g.json                   |
| MISMATCH    | S3.20   | SR 3      | held-out true negatives                                         | 17362      | 17382                  | sr03_l2g.json                   |
| MISMATCH    | S3.26   | SR 3      | CSs with more than one gene at L2G >= 0.5                       | 17463      | 14196                  | sr03_l2g.json                   |
| MISMATCH    | S3.27   | SR 3      | those as a share of qualified CSs (%)                           | 3.4        | 2.7                    | sr03_l2g.json                   |
| MISMATCH    | S3.28   | SR 3      | CSs with no gene at L2G >= 0.5                                  | 193523     | 196790                 | sr03_l2g.json                   |
| MISMATCH    | S3.29   | SR 3      | those as a share of qualified CSs (%)                           | 37.1       | 37.8                   | sr03_l2g.json                   |
| MISMATCH    | S3.35   | SR 3      | those as a share of prioritisations (%)                         | 13.0       | 12.1                   | sr03_l2g.json                   |
| MISMATCH    | S3.40   | SR 3      | genes associated when Orphanet, gene burden and eQTLs are added | 18950      | 18813                  | sr03_l2g.json                   |
| MISMATCH    | S3.41   | SR 3      | those as a share of protein-coding genes (%)                    | 94.1       | 93.5                   | sr03_l2g.json                   |
| MISMATCH    | S5.06   | SR 5      | of those, with a secondary CS that does (%)                     | 81.5       | 50.7                   | sr05_secondary_signals.json     |
| MISMATCH    | S6.07   | SR 6      | pleiotropic variants with concordance below 1                   | 1793       | 1051                   | sr06_variant_pleiotropy.json    |
| MISMATCH    | S6.16   | SR 6      | distinct coordinates those clusters occupy                      | 219        | 226                    | sr06_variant_pleiotropy.json    |
| MISMATCH    | S7.04   | SR 7      | mean discordance P in the joint model                           | 0.83       | 0.98                   | sr07_gps_discordance.json       |
| MISMATCH    | S7.05   | SR 7      | gPS beta with maximum discordance in the model                  | 0.25       | 0.28                   | sr07_gps_discordance.json       |
| MISMATCH    | S7.06   | SR 7      | gPS P with maximum discordance in the model                     | 5.9e-07    | 2.6562941591757134e-08 | sr07_gps_discordance.json       |
| MISMATCH    | S7.07   | SR 7      | maximum discordance P in the joint model                        | 0.14       | 0.54                   | sr07_gps_discordance.json       |
| MISMATCH    | S7.08   | SR 7      | maximum discordance beta, univariate                            | 1.82       | 1.64                   | sr07_gps_discordance.json       |
| MISMATCH    | S7.09   | SR 7      | maximum discordance P, univariate                               | 7.8e-07    | 7.115777933574665e-05  | sr07_gps_discordance.json       |
| MISMATCH    | S8.03   | SR 8      | OR with a random therapeutic-area intercept                     | 3.32       | 3.22                   | sr08_enrichment_bias.json       |
| MISMATCH    | S8.05   | SR 8      | variance of the therapeutic-area effect                         | 0.54       | 0.53                   | sr08_enrichment_bias.json       |
| MISMATCH    | S8.06   | SR 8      | standard deviation of the therapeutic-area effect               | 0.74       | 0.73                   | sr08_enrichment_bias.json       |
| BLOCKED     | R3.01   | Results 3 | L2G average precision on held-out test set                      | 0.81       |                        |                                 |
| BLOCKED     | R3.02   | Results 3 | L2G area under the curve                                        | 0.95       |                        |                                 |
| BLOCKED     | R3.05   | Results 3 | previous Open Targets model average precision                   | 0.65       |                        |                                 |
| BLOCKED     | R3.19   | Results 3 | OR for most LoF-constrained gene in CS neighbourhood            | 1.9        |                        |                                 |
| BLOCKED     | R5.23   | Results 5 | significantly enriched pathways (FDR < 0.05)                    | 312        |                        |                                 |
| BLOCKED     | R5.24   | Results 5 | pathways enriched in high-pleiotropy genes                      | 221        |                        |                                 |
| BLOCKED     | R5.25   | Results 5 | pathways enriched in low-pleiotropy genes                       | 91         |                        |                                 |
| BLOCKED     | S3.15   | SR 3      | average precision on the held-out set                           | 0.81       | 0.78                   | sr03_l2g.json                   |
| BLOCKED     | S3.16   | SR 3      | area under the curve on the held-out set                        | 0.95       | 0.91                   | sr03_l2g.json                   |
| PRECOMPUTED | S1.02   | SR 1      | studies with at least one credible set (%)                      | 27.3       |                        |                                 |
| PRECOMPUTED | S1.28   | SR 1      | GWAS Catalog studies excluded before fine-mapping (%)           | 30         |                        |                                 |
| PRECOMPUTED | S2.01   | SR 2      | overlaps significant by eCAVIAR (%)                             | 67         | 68.0                   | sr02_colocalisation.json        |
| PRECOMPUTED | S2.02   | SR 2      | overlaps significant by COLOC (%)                               | 79         | 79.3                   | sr02_colocalisation.json        |
| PRECOMPUTED | S3.42   | SR 3      | GWAS CSs novel against Open Targets Genetics 22.10              | 456323     |                        |                                 |
| PRECOMPUTED | S3.43   | SR 3      | those as a share of all GWAS CSs (%)                            | 58         |                        |                                 |
| PRECOMPUTED | S3.44   | SR 3      | GWAS CSs classified as previously known                         | 333130     |                        |                                 |
| PASS        | R1.01   | Results 1 | GWAS studies analysed                                           | 100526     | 100526                 | panoramic.json                  |
| PASS        | R1.02   | Results 1 | publications                                                    | 4250       | 4250                   | panoramic.json                  |
| PASS        | R1.03   | Results 1 | trait ontology terms                                            | 9280       | 9280                   | panoramic.json                  |
| PASS        | R1.04   | Results 1 | therapeutic areas                                               | 23         | 23                     | panoramic.json                  |
| PASS        | R1.05   | Results 1 | pre-2017 studies >10% non-European (%)                          | 23.2       | 23.2                   | panoramic.json                  |
| PASS        | R1.06   | Results 1 | by end-2024 studies >10% non-European (%)                       | 35.1       | 35.1                   | panoramic.json                  |
| PASS        | R1.07   | Results 1 | GWAS credible sets                                              | 789453     | 789453                 | panoramic.json                  |
| PASS        | R1.08   | Results 1 | studies with at least one CS                                    | 39282      | 39282                  | panoramic.json                  |
| PASS        | R1.09   | Results 1 | qualified CSs                                                   | 520975     | 520975                 | panoramic.json                  |
| PASS        | R1.10   | Results 1 | qualified disease CSs                                           | 70618      | 70618                  | panoramic.json                  |
| PASS        | R1.11   | Results 1 | qualified measurement CSs                                       | 450357     | 450357                 | panoramic.json                  |
| PASS        | R1.12   | Results 1 | unique variants in CSs                                          | 2024916    | 2024916                | panoramic.json                  |
| PASS        | R1.13   | Results 1 | lead variants                                                   | 211597     | 211597                 | panoramic.json                  |
| PASS        | R1.14   | Results 1 | lead variants with PIP >= 0.9                                   | 49772      | 49772                  | panoramic.json                  |
| PASS        | R1.15   | Results 1 | molQTL CSs used for colocalisation                              | 2044305    | 2044305                | panoramic.json                  |
| PASS        | R1.16   | Results 1 | tissues or cell types                                           | 98         | 98                     | panoramic.json                  |
| PASS        | R1.17   | Results 1 | CS-gene prioritisations                                         | 523409     | 523409                 | panoramic.json                  |
| PASS        | R1.18   | Results 1 | gene-disease pairs                                              | 36858      | 36858                  | panoramic.json                  |
| PASS        | R1.19   | Results 1 | gene-measurement pairs                                          | 150360     | 150360                 | panoramic.json                  |
| PASS        | R1.20   | Results 1 | unique genes implicated                                         | 15641      | 15641                  | panoramic.json                  |
| PASS        | R1.21   | Results 1 | disease-associated genes                                        | 8285       | 8285                   | panoramic.json                  |
| PASS        | R1.22   | Results 1 | measurement-associated genes                                    | 15160      | 15160                  | panoramic.json                  |
| PASS        | R1.23   | Results 1 | diseases covered                                                | 1394       | 1394                   | panoramic.json                  |
| PASS        | R1.24   | Results 1 | measurements covered                                            | 3412       | 3412                   | panoramic.json                  |
| PASS        | R1.25   | Results 1 | protein-coding genes associated (%)                             | 77.9       | 77.9                   | panoramic.json                  |
| PASS        | R1.26   | Results 1 | genes not from EUR common studies by 2024                       | 2462       | 2462                   | panoramic.json                  |
| PASS        | R1.27   | Results 1 | disease-associated genes by 2024 (denominator)                  | 8129       | 8129                   | panoramic.json                  |
| PASS        | R1.28   | Results 1 | genes from non-EUR studies                                      | 1829       | 1829                   | panoramic.json                  |
| PASS        | R1.29   | Results 1 | genes from mixed studies                                        | 633        | 633                    | panoramic.json                  |
| PASS        | R1.30   | Results 1 | gene-disease pairs not from EUR common                          | 16384      | 16384                  | panoramic.json                  |
| PASS        | R1.31   | Results 1 | gene-disease pairs by 2024 (denominator)                        | 34905      | 34905                  | panoramic.json                  |
| PASS        | R1.32   | Results 1 | gene-disease pairs from non-EUR                                 | 12285      | 12285                  | panoramic.json                  |
| PASS        | R1.33   | Results 1 | gene-disease pairs from mixed                                   | 4099       | 4099                   | panoramic.json                  |
| PASS        | R1.34   | Results 1 | IRR non-EUR vs EUR                                              | 1.69       | 1.69                   | sr12_discovery_regression.json  |
| PASS        | R1.35   | Results 1 | IRR mixed vs EUR                                                | 0.97       | 0.97                   | sr12_discovery_regression.json  |
| PASS        | R1.36   | Results 1 | IRR per tenfold effective sample size                           | 3.67       | 3.67                   | sr12_discovery_regression.json  |
| PASS        | R1.37   | Results 1 | genes associated with 2+ diseases                               | 5314       | 5314                   | panoramic.json                  |
| PASS        | R1.38   | Results 1 | genes associated with >1 therapeutic area                       | 4743       | 4743                   | panoramic.json                  |
| PASS        | R2.02   | Results 2 | enhancer-overlapping share in GWAS disease loci (%)             | 13.8       | 13.8                   | selective_pressures.json        |
| PASS        | R2.03   | Results 2 | promoter-overlapping share in cis-pQTLs (%)                     | 9.1        | 9.1                    | selective_pressures.json        |
| PASS        | R2.04   | Results 2 | promoter-overlapping share in GWAS disease loci (%)             | 2.9        | 2.9                    | selective_pressures.json        |
| PASS        | R2.05   | Results 2 | replicated lead variants intragenic or PAV (%)                  | 63         | 63                     | selective_pressures.json        |
| PASS        | R3.03   | Results 3 | L2G recall at score >= 0.5                                      | 0.65       | 0.65                   | sr04_l2g_vs_naive.json          |
| PASS        | R3.04   | Results 3 | CS-protein coding gene pairs scored                             | 7066749    | 7066749                | colocalisation_l2g.json         |
| PASS        | R3.06   | Results 3 | L2G FDR at score >= 0.5 (%)                                     | 11.5       | 11.5                   | sr04_l2g_vs_naive.json          |
| PASS        | R3.07   | Results 3 | nearest-gene FDR (%)                                            | 27         | 27.0                   | sr04_l2g_vs_naive.json          |
| PASS        | R3.08   | Results 3 | prioritised genes lacking PAV/molQTL support in 2015 (%)        | 49         | 49.0                   | colocalisation_l2g.json         |
| PASS        | R3.10   | Results 3 | prioritised genes with eQTL colocalisation (%)                  | 36.7       | 36.7                   | colocalisation_l2g.json         |
| PASS        | R3.11   | Results 3 | prioritised genes with pQTL colocalisation (%)                  | 5.7        | 5.7                    | colocalisation_l2g.json         |
| PASS        | R3.13   | Results 3 | eQTL colocalisation sensitivity (%)                             | 21.8       | 21.8                   | sr04_l2g_vs_naive.json          |
| PASS        | R3.14   | Results 3 | eQTL colocalisation FDR (%)                                     | 65.1       | 65.1                   | sr04_l2g_vs_naive.json          |
| PASS        | R3.15   | Results 3 | CS-gene assignments that are nearest to TSS (%)                 | 81.2       | 81.2                   | colocalisation_l2g.json         |
| PASS        | R3.16   | Results 3 | nearest-gene assignments with no PAV or coloc support (%)       | 46.1       | 46.1                   | colocalisation_l2g.json         |
| PASS        | R3.18   | Results 3 | MST1 L2G score                                                  | 0.86       | 0.86                   | colocalisation_l2g.json         |
| PASS        | R4.01   | Results 4 | independent colocalisation clusters                             | 20041      | 20041                  | variant_pleiotropy.json         |
| PASS        | R4.02   | Results 4 | clusters with more than one lead variant                        | 5595       | 5595                   | variant_pleiotropy.json         |
| PASS        | R4.03   | Results 4 | clusters linked to multiple diseases                            | 6617       | 6617                   | variant_pleiotropy.json         |
| PASS        | R4.04   | Results 4 | maximum diseases per cluster                                    | 120        | 120                    | variant_pleiotropy.json         |
| PASS        | R4.05   | Results 4 | mean diseases per cluster                                       | 2.14       | 2.14                   | variant_pleiotropy.json         |
| PASS        | R4.22   | Results 4 | APOE 19_44908684_T_C therapeutic areas                          | 15         | 15                     | variant_pleiotropy.json         |
| PASS        | R5.01   | Results 5 | disease genes with gPS > 1                                      | 5314       | 5314                   | gene_pleiotropy.json            |
| PASS        | R5.02   | Results 5 | mean gPS                                                        | 4.45       | 4.45                   | gene_pleiotropy.json            |
| PASS        | R5.03   | Results 5 | maximum gPS                                                     | 148        | 148                    | gene_pleiotropy.json            |
| PASS        | R5.04   | Results 5 | genes linked to multiple therapeutic areas                      | 4743       | 4743                   | gene_pleiotropy.json            |
| PASS        | R5.05   | Results 5 | mean therapeutic areas per gene                                 | 2.53       | 2.53                   | gene_pleiotropy.json            |
| PASS        | R5.06   | Results 5 | maximum therapeutic areas per gene                              | 21         | 21                     | gene_pleiotropy.json            |
| PASS        | R5.07   | Results 5 | Spearman rho gPS vs therapeutic area count                      | 0.92       | 0.92                   | gene_pleiotropy.json            |
| PASS        | R5.08   | Results 5 | FTO gPS                                                         | 126        | 126                    | gene_pleiotropy.json            |
| PASS        | R5.09   | Results 5 | APOE gPS                                                        | 107        | 107                    | gene_pleiotropy.json            |
| PASS        | R5.10   | Results 5 | ABO gPS                                                         | 105        | 105                    | gene_pleiotropy.json            |
| PASS        | R5.11   | Results 5 | CDKN2B gPS                                                      | 148        | 148                    | gene_pleiotropy.json            |
| PASS        | R5.12   | Results 5 | CDKN2B therapeutic areas                                        | 21         | 21                     | gene_pleiotropy.json            |
| PASS        | R5.13   | Results 5 | Spearman rho variants per gene vs gPS                           | 0.87       | 0.87                   | gene_pleiotropy.json            |
| PASS        | R5.14   | Results 5 | gPS joint model Pearson R2                                      | 0.15       | 0.1506314030361559     | gene_pleiotropy.json            |
| PASS        | R5.15   | Results 5 | gPS univariate beta max sample size                             | 1.82       | 1.823882722925956      | gene_pleiotropy.json            |
| PASS        | R5.18   | Results 5 | gPS univariate beta pathway count                               | 3.45       | 3.4453201549474812     | gene_pleiotropy.json            |
| PASS        | R5.19   | Results 5 | gPS univariate beta gene length                                 | 2.17       | 2.168946084659057      | gene_pleiotropy.json            |
| PASS        | R5.20   | Results 5 | gPS univariate beta PAV association                             | 0.53       | 0.5293316780527356     | gene_pleiotropy.json            |
| PASS        | R5.21   | Results 5 | gPS univariate beta tissue specificity                          | -0.11      | -0.10513881341014132   | gene_pleiotropy.json            |
| PASS        | R5.22   | Results 5 | Pearson r LoF vs missense constraint                            | 0.628      | 0.627                  | gene_pleiotropy.json            |
| PASS        | R5.26   | Results 5 | gene sets tested                                                | 21         | 21                     | gene_pleiotropy.json            |
| PASS        | R5.27   | Results 5 | gene sets significantly associated with higher gPS              | 10         | 10                     | gene_pleiotropy.json            |
| PASS        | R5.28   | Results 5 | OR cancer driver genes per doubling of gPS                      | 1.49       | 1.498403014446467      | gene_pleiotropy.json            |
| PASS        | R5.29   | Results 5 | OR mouse knockout-lethal homologs                               | 1.30       | 1.298431308959641      | gene_pleiotropy.json            |
| PASS        | R5.30   | Results 5 | OR developmental disorder panel genes                           | 1.32       | 1.3252845063271235     | gene_pleiotropy.json            |
| PASS        | R5.31   | Results 5 | OR safety-terminated trial targets                              | 1.31       | 1.3036905573738717     | gene_pleiotropy.json            |
| PASS        | R5.32   | Results 5 | OR withdrawn drug targets                                       | 1.08       | 1.0795732587780944     | gene_pleiotropy.json            |
| PASS        | R5.33   | Results 5 | withdrawn drug target gene set size                             | 166        | 166                    | gene_pleiotropy.json            |
| PASS        | R5.34   | Results 5 | OR known safety event targets                                   | 1.13       | 1.123599951957629      | gene_pleiotropy.json            |
| PASS        | R5.35   | Results 5 | known safety event gene set size                                | 214        | 214                    | gene_pleiotropy.json            |
| PASS        | R5.36   | Results 5 | OR LoF-constrained genes Q4                                     | 1.23       | 1.2353438993640622     | gene_pleiotropy.json            |
| PASS        | R5.37   | Results 5 | OR low-constraint genes Q1                                      | 0.85       | 0.8547520776244846     | gene_pleiotropy.json            |
| PASS        | R5.38   | Results 5 | OR human knockout genes                                         | 0.89       | 0.887662441222951      | gene_pleiotropy.json            |
| PASS        | R6.01   | Results 6 | approved target-indication pairs with GWAS support              | 242        | 242                    | therapeutic_success.json        |
| PASS        | R6.02   | Results 6 | overall genetic support OR                                      | 3.62       | 3.62                   | therapeutic_success.json        |
| PASS        | R6.03   | Results 6 | overall relative success                                        | 2.76       | 2.76                   | therapeutic_success.json        |
| PASS        | R6.04   | Results 6 | OR after mixed-effects adjustment                               | 3.14       | 3.14                   | sr08_enrichment_bias.json       |
| PASS        | R6.05   | Results 6 | OR rare-variant associations                                    | 7.0        | 6.994051439745637      | therapeutic_success.json        |
| PASS        | R6.06   | Results 6 | OR common-variant associations                                  | 3.4        | 3.3954651611381843     | therapeutic_success.json        |
| PASS        | R6.07   | Results 6 | P rare vs common                                                | 0.0077     | 0.0077                 | therapeutic_success.json        |
| PASS        | R6.10   | Results 6 | OR ClinVar/ClinGen                                              | 5.1        | 5.1                    | therapeutic_success.json        |
| PASS        | R6.11   | Results 6 | OR UniProt                                                      | 5.0        | 5.0                    | therapeutic_success.json        |
| PASS        | R6.12   | Results 6 | OR Genomics England PanelApp                                    | 4.5        | 4.5                    | therapeutic_success.json        |
| PASS        | R6.13   | Results 6 | OR PAV-supported associations                                   | 6.0        | 6.048323445762209      | therapeutic_success.json        |
| PASS        | R6.14   | Results 6 | OR non-PAV associations                                         | 3.1        | 3.092428147282449      | therapeutic_success.json        |
| PASS        | R6.15   | Results 6 | P PAV vs non-PAV                                                | 0.0002     | 0.0002                 | therapeutic_success.json        |
| PASS        | R6.16   | Results 6 | OR gene-based analyses                                          | 7.2        | 7.2                    | therapeutic_success.json        |
| PASS        | R6.17   | Results 6 | OR large effect                                                 | 4.6        | 4.6282475044622196     | therapeutic_success.json        |
| PASS        | R6.18   | Results 6 | OR small effect                                                 | 3.5        | 3.4730186783176276     | therapeutic_success.json        |
| PASS        | R6.19   | Results 6 | OR gPS <= 5                                                     | 4.8        | 4.798286448368984      | therapeutic_success.json        |
| PASS        | R6.20   | Results 6 | OR gPS >= 10                                                    | 3.0        | 2.9677312242353167     | therapeutic_success.json        |
| PASS        | R6.21   | Results 6 | P gPS low vs high                                               | 0.008      | 0.008                  | therapeutic_success.json        |
| PASS        | R6.22   | Results 6 | OR single therapeutic area                                      | 4.3        | 4.2907160793554455     | therapeutic_success.json        |
| PASS        | R6.23   | Results 6 | OR six or more therapeutic areas                                | 2.9        | 2.8881755914500533     | therapeutic_success.json        |
| PASS        | R6.24   | Results 6 | P TA count low vs high                                          | 0.18       | 0.18                   | therapeutic_success.json        |
| PASS        | R6.27   | Results 6 | gene-disease associations meeting PAV + 2-5 TA definition       | 2734       | 2734                   | therapeutic_success.json        |
| PASS        | R6.28   | Results 6 | OR PAV + 2-5 therapeutic areas                                  | 10.3       | 10.3                   | therapeutic_success.json        |
| PASS        | R6.29   | Results 6 | RS PAV + 2-5 therapeutic areas                                  | 4.8        | 4.8                    | therapeutic_success.json        |
| PASS        | R6.30   | Results 6 | approved pairs also meeting the strict definition               | 51         | 51                     | therapeutic_success.json        |
| PASS        | R6.31   | Results 6 | that subset as a share of 242 (%)                               | 21.1       | 21.1                   | therapeutic_success.json        |
| PASS        | S1.01   | SR 1      | binary traits among GWAS with a credible set (%)                | 20.24      | 20.24                  | sr01_finemapping_catalogue.json |
| PASS        | S1.03   | SR 1      | GWAS with a CS not reaching 90% NFE (%)                         | 32         | 31.9                   | sr01_finemapping_catalogue.json |
| PASS        | S1.04   | SR 1      | of those, non-EUR (%)                                           | 18.2       | 18.2                   | sr01_finemapping_catalogue.json |
| PASS        | S1.05   | SR 1      | of those, mixed (%)                                             | 13.7       | 13.7                   | sr01_finemapping_catalogue.json |
| PASS        | S1.06   | SR 1      | within non-EUR, East Asian (%)                                  | 50.0       | 50.0                   | sr01_finemapping_catalogue.json |
| PASS        | S1.07   | SR 1      | within non-EUR, African (%)                                     | 31.5       | 31.5                   | sr01_finemapping_catalogue.json |
| PASS        | S1.08   | SR 1      | within non-EUR, Finnish (%)                                     | 9.8        | 9.8                    | sr01_finemapping_catalogue.json |
| PASS        | S1.09   | SR 1      | within non-EUR, American (%)                                    | 8.7        | 8.7                    | sr01_finemapping_catalogue.json |
| PASS        | S1.10   | SR 1      | mixed studies predominantly NFE (%)                             | 82.7       | 82.7                   | sr01_finemapping_catalogue.json |
| PASS        | S1.11   | SR 1      | earliest publication year                                       | 2006       | 2006                   | sr01_finemapping_catalogue.json |
| PASS        | S1.12   | SR 1      | mean GWAS credible set size                                     | 24.61      | 24.61                  | sr01_finemapping_catalogue.json |
| PASS        | S1.13   | SR 1      | median GWAS credible set size                                   | 5          | 5                      | sr01_finemapping_catalogue.json |
| PASS        | S1.14   | SR 1      | single-variant resolution, FinnGen (%)                          | 13.5       | 13.5                   | sr01_finemapping_catalogue.json |
| PASS        | S1.15   | SR 1      | single-variant resolution, GWAS Catalog sumstats (%)            | 40.1       | 40.1                   | sr01_finemapping_catalogue.json |
| PASS        | S1.17   | SR 1      | unique genes covered by molQTL credible sets                    | 29342      | 29342                  | sr01_finemapping_catalogue.json |
| PASS        | S1.18   | SR 1      | eQTL credible sets                                              | 1402222    | 1402222                | sr01_finemapping_catalogue.json |
| PASS        | S1.19   | SR 1      | pQTL credible sets                                              | 33731      | 33731                  | sr01_finemapping_catalogue.json |
| PASS        | S1.20   | SR 1      | replicated GWAS credible sets                                   | 263705     | 263705                 | sr01_finemapping_catalogue.json |
| PASS        | S1.21   | SR 1      | replicated GWAS credible sets (%)                               | 33.4       | 33.4                   | sr01_finemapping_catalogue.json |
| PASS        | S1.22   | SR 1      | replicated molQTL credible sets                                 | 1461445    | 1461445                | sr01_finemapping_catalogue.json |
| PASS        | S1.23   | SR 1      | replicated molQTL credible sets (%)                             | 71.5       | 71.5                   | sr01_finemapping_catalogue.json |
| PASS        | S1.24   | SR 1      | qualified measurement studies                                   | 61885      | 61885                  | sr01_finemapping_catalogue.json |
| PASS        | S1.25   | SR 1      | qualified disease studies                                       | 15730      | 15730                  | sr01_finemapping_catalogue.json |
| PASS        | S1.26   | SR 1      | qualified credible sets with a rare lead variant                | 15311      | 15311                  | sr01_finemapping_catalogue.json |
| PASS        | S1.27   | SR 1      | those as a share of qualified credible sets (%)                 | 2.94       | 2.94                   | sr01_finemapping_catalogue.json |
| PASS        | S10.01  | SR 10     | target-indication pairs with genetic support                    | 18480      | 18480                  | sr10_phase_transitions.json     |
| PASS        | S10.02  | SR 10     | pairs in group 1 TA (therapeutic areas)                         | 6578       | 6578                   | sr10_phase_transitions.json     |
| PASS        | S10.03  | SR 10     | pairs in group 2–5 TAs (therapeutic areas)                      | 9705       | 9705                   | sr10_phase_transitions.json     |
| PASS        | S10.04  | SR 10     | pairs in group ≥6 TAs (therapeutic areas)                       | 2197       | 2197                   | sr10_phase_transitions.json     |
| PASS        | S10.05  | SR 10     | Phase I→II, 1 TA: pairs entering                                | 6578       | 6578                   | sr10_phase_transitions.json     |
| PASS        | S10.06  | SR 10     | Phase I→II, 1 TA: success (%)                                   | 82.9       | 82.9                   | sr10_phase_transitions.json     |
| PASS        | S10.07  | SR 10     | Phase I→II, 2–5 TAs: pairs entering                             | 9705       | 9705                   | sr10_phase_transitions.json     |
| PASS        | S10.08  | SR 10     | Phase I→II, 2–5 TAs: success (%)                                | 84.2       | 84.2                   | sr10_phase_transitions.json     |
| PASS        | S10.09  | SR 10     | Phase I→II, ≥6 TAs: pairs entering                              | 2197       | 2197                   | sr10_phase_transitions.json     |
| PASS        | S10.10  | SR 10     | Phase I→II, ≥6 TAs: success (%)                                 | 82.6       | 82.6                   | sr10_phase_transitions.json     |
| PASS        | S10.11  | SR 10     | Phase I→II, med_vs_low (therapeutic areas): risk ratio          | 1.02       | 1.02                   | sr10_phase_transitions.json     |
| PASS        | S10.12  | SR 10     | Phase I→II, high_vs_low (therapeutic areas): risk ratio         | 1.0        | 1.0                    | sr10_phase_transitions.json     |
| PASS        | S10.13  | SR 10     | Phase I→II, high_vs_med (therapeutic areas): risk ratio         | 0.98       | 0.98                   | sr10_phase_transitions.json     |
| PASS        | S10.14  | SR 10     | Phase II→III, 1 TA: pairs entering                              | 5453       | 5453                   | sr10_phase_transitions.json     |
| PASS        | S10.15  | SR 10     | Phase II→III, 1 TA: success (%)                                 | 57.0       | 57.0                   | sr10_phase_transitions.json     |
| PASS        | S10.16  | SR 10     | Phase II→III, 2–5 TAs: pairs entering                           | 8173       | 8173                   | sr10_phase_transitions.json     |
| PASS        | S10.17  | SR 10     | Phase II→III, 2–5 TAs: success (%)                              | 53.5       | 53.5                   | sr10_phase_transitions.json     |
| PASS        | S10.18  | SR 10     | Phase II→III, ≥6 TAs: pairs entering                            | 1815       | 1815                   | sr10_phase_transitions.json     |
| PASS        | S10.19  | SR 10     | Phase II→III, ≥6 TAs: success (%)                               | 49.8       | 49.8                   | sr10_phase_transitions.json     |
| PASS        | S10.20  | SR 10     | Phase II→III, med_vs_low (therapeutic areas): risk ratio        | 0.94       | 0.94                   | sr10_phase_transitions.json     |
| PASS        | S10.21  | SR 10     | Phase II→III, high_vs_low (therapeutic areas): risk ratio       | 0.87       | 0.87                   | sr10_phase_transitions.json     |
| PASS        | S10.22  | SR 10     | Phase II→III, high_vs_med (therapeutic areas): risk ratio       | 0.93       | 0.93                   | sr10_phase_transitions.json     |
| PASS        | S10.23  | SR 10     | Phase III→approval, 1 TA: pairs entering                        | 3110       | 3110                   | sr10_phase_transitions.json     |
| PASS        | S10.24  | SR 10     | Phase III→approval, 1 TA: success (%)                           | 29.4       | 29.4                   | sr10_phase_transitions.json     |
| PASS        | S10.25  | SR 10     | Phase III→approval, 2–5 TAs: pairs entering                     | 4376       | 4376                   | sr10_phase_transitions.json     |
| PASS        | S10.26  | SR 10     | Phase III→approval, 2–5 TAs: success (%)                        | 31.2       | 31.2                   | sr10_phase_transitions.json     |
| PASS        | S10.27  | SR 10     | Phase III→approval, ≥6 TAs: pairs entering                      | 903        | 903                    | sr10_phase_transitions.json     |
| PASS        | S10.28  | SR 10     | Phase III→approval, ≥6 TAs: success (%)                         | 30.6       | 30.6                   | sr10_phase_transitions.json     |
| PASS        | S10.29  | SR 10     | Phase III→approval, med_vs_low (therapeutic areas): risk ratio  | 1.06       | 1.06                   | sr10_phase_transitions.json     |
| PASS        | S10.30  | SR 10     | Phase III→approval, high_vs_low (therapeutic areas): risk ratio | 1.04       | 1.04                   | sr10_phase_transitions.json     |
| PASS        | S10.31  | SR 10     | Phase III→approval, high_vs_med (therapeutic areas): risk ratio | 0.98       | 0.98                   | sr10_phase_transitions.json     |
| PASS        | S10.32  | SR 10     | pairs in group gPS 1 (gPS)                                      | 5724       | 5724                   | sr10_phase_transitions.json     |
| PASS        | S10.33  | SR 10     | pairs in group gPS 2–9 (gPS)                                    | 9542       | 9542                   | sr10_phase_transitions.json     |
| PASS        | S10.34  | SR 10     | pairs in group gPS ≥10 (gPS)                                    | 3214       | 3214                   | sr10_phase_transitions.json     |
| PASS        | S10.35  | SR 10     | Phase I→II, gPS 1: pairs entering                               | 5724       | 5724                   | sr10_phase_transitions.json     |
| PASS        | S10.36  | SR 10     | Phase I→II, gPS 1: success (%)                                  | 82.3       | 82.3                   | sr10_phase_transitions.json     |
| PASS        | S10.37  | SR 10     | Phase I→II, gPS 2–9: pairs entering                             | 9542       | 9542                   | sr10_phase_transitions.json     |
| PASS        | S10.38  | SR 10     | Phase I→II, gPS 2–9: success (%)                                | 84.4       | 84.4                   | sr10_phase_transitions.json     |
| PASS        | S10.39  | SR 10     | Phase I→II, gPS ≥10: pairs entering                             | 3214       | 3214                   | sr10_phase_transitions.json     |
| PASS        | S10.40  | SR 10     | Phase I→II, gPS ≥10: success (%)                                | 83.4       | 83.4                   | sr10_phase_transitions.json     |
| PASS        | S10.41  | SR 10     | Phase I→II, med_vs_low (gPS): risk ratio                        | 1.03       | 1.03                   | sr10_phase_transitions.json     |
| PASS        | S10.42  | SR 10     | Phase I→II, high_vs_low (gPS): risk ratio                       | 1.01       | 1.01                   | sr10_phase_transitions.json     |
| PASS        | S10.43  | SR 10     | Phase I→II, high_vs_med (gPS): risk ratio                       | 0.99       | 0.99                   | sr10_phase_transitions.json     |
| PASS        | S10.44  | SR 10     | Phase II→III, gPS 1: pairs entering                             | 4708       | 4708                   | sr10_phase_transitions.json     |
| PASS        | S10.45  | SR 10     | Phase II→III, gPS 1: success (%)                                | 56.7       | 56.7                   | sr10_phase_transitions.json     |
| PASS        | S10.46  | SR 10     | Phase II→III, gPS 2–9: pairs entering                           | 8051       | 8051                   | sr10_phase_transitions.json     |
| PASS        | S10.47  | SR 10     | Phase II→III, gPS 2–9: success (%)                              | 53.8       | 53.8                   | sr10_phase_transitions.json     |
| PASS        | S10.48  | SR 10     | Phase II→III, gPS ≥10: pairs entering                           | 2682       | 2682                   | sr10_phase_transitions.json     |
| PASS        | S10.49  | SR 10     | Phase II→III, gPS ≥10: success (%)                              | 51.6       | 51.6                   | sr10_phase_transitions.json     |
| PASS        | S10.50  | SR 10     | Phase II→III, med_vs_low (gPS): risk ratio                      | 0.95       | 0.95                   | sr10_phase_transitions.json     |
| PASS        | S10.51  | SR 10     | Phase II→III, high_vs_low (gPS): risk ratio                     | 0.91       | 0.91                   | sr10_phase_transitions.json     |
| PASS        | S10.52  | SR 10     | Phase II→III, high_vs_med (gPS): risk ratio                     | 0.96       | 0.96                   | sr10_phase_transitions.json     |
| PASS        | S10.53  | SR 10     | Phase III→approval, gPS 1: pairs entering                       | 2671       | 2671                   | sr10_phase_transitions.json     |
| PASS        | S10.54  | SR 10     | Phase III→approval, gPS 1: success (%)                          | 29.0       | 29.0                   | sr10_phase_transitions.json     |
| PASS        | S10.55  | SR 10     | Phase III→approval, gPS 2–9: pairs entering                     | 4335       | 4335                   | sr10_phase_transitions.json     |
| PASS        | S10.56  | SR 10     | Phase III→approval, gPS 2–9: success (%)                        | 31.7       | 31.7                   | sr10_phase_transitions.json     |
| PASS        | S10.57  | SR 10     | Phase III→approval, gPS ≥10: pairs entering                     | 1383       | 1383                   | sr10_phase_transitions.json     |
| PASS        | S10.58  | SR 10     | Phase III→approval, gPS ≥10: success (%)                        | 29.4       | 29.4                   | sr10_phase_transitions.json     |
| PASS        | S10.59  | SR 10     | Phase III→approval, med_vs_low (gPS): risk ratio                | 1.09       | 1.09                   | sr10_phase_transitions.json     |
| PASS        | S10.60  | SR 10     | Phase III→approval, high_vs_low (gPS): risk ratio               | 1.01       | 1.01                   | sr10_phase_transitions.json     |
| PASS        | S10.61  | SR 10     | Phase III→approval, high_vs_med (gPS): risk ratio               | 0.93       | 0.93                   | sr10_phase_transitions.json     |
| PASS        | S11.01  | SR 11     | pairs meeting the combined criterion                            | 87         | 87                     | sr11_criterion.json             |
| PASS        | S11.02  | SR 11     | of those, approved                                              | 51         | 51                     | sr11_criterion.json             |
| PASS        | S11.03  | SR 11     | that as a share of the 242 approved supported pairs (%)         | 21.1       | 21.1                   | sr11_criterion.json             |
| PASS        | S11.04  | SR 11     | OR for PAV support                                              | 6.05       | 6.05                   | sr11_criterion.json             |
| PASS        | S11.05  | SR 11     | OR for non-PAV support                                          | 3.09       | 3.09                   | sr11_criterion.json             |
| PASS        | S11.06  | SR 11     | P for the difference between them                               | 2.4e-04    | 0.00024409528821107924 | sr11_criterion.json             |
| PASS        | S11.07  | SR 11     | reportable PAV therapeutic-area windows                         | 64         | 64                     | sr11_criterion.json             |
| PASS        | S11.08  | SR 11     | rank of the reported 2-5 window                                 | 4          | 4                      | sr11_criterion.json             |
| PASS        | S11.09  | SR 11     | windows whose OR falls inside the reported CI                   | 32         | 32                     | sr11_criterion.json             |
| PASS        | S11.10  | SR 11     | windows whose CI excludes the all-GWAS baseline                 | 45         | 45                     | sr11_criterion.json             |
| PASS        | S11.100 | SR 11     | power against an attenuated effect (%)                          | 43         | 36.0                   | sr11_external.json              |
| PASS        | S11.101 | SR 11     | power against a ChEMBL-strength effect (%)                      | 99         | 99.0                   | sr11_external.json              |
| PASS        | S11.102 | SR 11     | attenuation factor on the log-odds scale                        | 0.392      | 0.392                  | sr11_external.json              |
| PASS        | S11.103 | SR 11     | Pharmaprojects quadratic coefficient                            | -0.042     | -0.042                 | sr11_external.json              |
| PASS        | S11.104 | SR 11     | its CI low                                                      | -0.164     | -0.164                 | sr11_external.json              |
| PASS        | S11.105 | SR 11     | its CI high                                                     | 0.08       | 0.08                   | sr11_external.json              |
| PASS        | S11.108 | SR 11     | ceiling, ChEMBL PAV ratio                                       | 3.33       | 3.33                   | sr11_external.json              |
| PASS        | S11.109 | SR 11     | ceiling, ChEMBL PAV P                                           | 0.00048    | 0.00048036000049630665 | sr11_external.json              |
| PASS        | S11.11  | SR 11     | windows with a higher point estimate                            | 3          | 3                      | sr11_criterion.json             |
| PASS        | S11.110 | SR 11     | ceiling, ChEMBL PAV low supported pairs                         | 87         | 87                     | sr11_external.json              |
| PASS        | S11.111 | SR 11     | ceiling, ChEMBL PAV low approved pairs                          | 51         | 51                     | sr11_external.json              |
| PASS        | S11.112 | SR 11     | ceiling, ChEMBL PAV high supported pairs                        | 67         | 67                     | sr11_external.json              |
| PASS        | S11.113 | SR 11     | ceiling, ChEMBL PAV high approved pairs                         | 20         | 20                     | sr11_external.json              |
| PASS        | S11.116 | SR 11     | ceiling, Pharmaprojects PAV ratio                               | 4.14       | 4.14                   | sr11_external.json              |
| PASS        | S11.117 | SR 11     | ceiling, Pharmaprojects PAV P                                   | 0.0014     | 0.0014092811320908895  | sr11_external.json              |
| PASS        | S11.118 | SR 11     | ceiling, Pharmaprojects PAV low supported pairs                 | 60         | 60                     | sr11_external.json              |
| PASS        | S11.119 | SR 11     | ceiling, Pharmaprojects PAV low approved pairs                  | 23         | 23                     | sr11_external.json              |
| PASS        | S11.12  | SR 11     | OR at exactly 2 therapeutic areas                               | 18.01      | 18.01                  | sr11_criterion.json             |
| PASS        | S11.120 | SR 11     | ceiling, Pharmaprojects PAV high supported pairs                | 69         | 69                     | sr11_external.json              |
| PASS        | S11.121 | SR 11     | ceiling, Pharmaprojects PAV high approved pairs                 | 9          | 9                      | sr11_external.json              |
| PASS        | S11.122 | SR 11     | ceiling, ChEMBL any-support low OR                              | 4.01       | 4.01                   | sr11_external.json              |
| PASS        | S11.123 | SR 11     | ceiling, ChEMBL any-support high OR                             | 2.97       | 2.97                   | sr11_external.json              |
| PASS        | S11.124 | SR 11     | ceiling, ChEMBL any-support ratio                               | 1.35       | 1.35                   | sr11_external.json              |
| PASS        | S11.125 | SR 11     | ceiling, ChEMBL any-support P                                   | 0.073      | 0.07328518618834312    | sr11_external.json              |
| PASS        | S11.126 | SR 11     | ceiling, ChEMBL any-support low supported pairs                 | 398        | 398                    | sr11_external.json              |
| PASS        | S11.127 | SR 11     | ceiling, ChEMBL any-support low approved pairs                  | 139        | 139                    | sr11_external.json              |
| PASS        | S11.128 | SR 11     | ceiling, ChEMBL any-support high supported pairs                | 285        | 285                    | sr11_external.json              |
| PASS        | S11.129 | SR 11     | ceiling, ChEMBL any-support high approved pairs                 | 81         | 81                     | sr11_external.json              |
| PASS        | S11.13  | SR 11     | approved pairs behind it                                        | 10         | 10                     | sr11_criterion.json             |
| PASS        | S11.130 | SR 11     | ceiling, Pharmaprojects any-support low OR                      | 1.88       | 1.88                   | sr11_external.json              |
| PASS        | S11.131 | SR 11     | ceiling, Pharmaprojects any-support high OR                     | 1.48       | 1.48                   | sr11_external.json              |
| PASS        | S11.132 | SR 11     | ceiling, Pharmaprojects any-support ratio                       | 1.27       | 1.27                   | sr11_external.json              |
| PASS        | S11.133 | SR 11     | ceiling, Pharmaprojects any-support P                           | 0.34       | 0.3399244552717705     | sr11_external.json              |
| PASS        | S11.134 | SR 11     | ceiling, Pharmaprojects any-support low supported pairs         | 202        | 202                    | sr11_external.json              |
| PASS        | S11.135 | SR 11     | ceiling, Pharmaprojects any-support low approved pairs          | 41         | 41                     | sr11_external.json              |
| PASS        | S11.136 | SR 11     | ceiling, Pharmaprojects any-support high supported pairs        | 233        | 233                    | sr11_external.json              |
| PASS        | S11.137 | SR 11     | ceiling, Pharmaprojects any-support high approved pairs         | 39         | 39                     | sr11_external.json              |
| PASS        | S11.138 | SR 11     | interaction OR in ChEMBL                                        | 3.28       | 3.28                   | sr11_external.json              |
| PASS        | S11.139 | SR 11     | its CI low                                                      | 1.51       | 1.51                   | sr11_external.json              |
| PASS        | S11.14  | SR 11     | OR for the 2-3 window                                           | 10.82      | 10.82                  | sr11_criterion.json             |
| PASS        | S11.140 | SR 11     | its CI high                                                     | 7.13       | 7.13                   | sr11_external.json              |
| PASS        | S11.141 | SR 11     | its likelihood-ratio P                                          | 0.0024     | 0.002383280159621927   | sr11_external.json              |
| PASS        | S11.142 | SR 11     | its permutation P                                               | 0.0018     | 0.0017998200179982     | sr11_external.json              |
| PASS        | S11.143 | SR 11     | supported pairs in the ChEMBL model                             | 683        | 683                    | sr11_external.json              |
| PASS        | S11.144 | SR 11     | approved pairs among them                                       | 220        | 220                    | sr11_external.json              |
| PASS        | S11.145 | SR 11     | ChEMBL approval rate, 2-5 TAs without a PAV (%)                 | 28.3       | 28.3                   | sr11_external.json              |
| PASS        | S11.146 | SR 11     | ChEMBL approval rate, 2-5 TAs with a PAV (%)                    | 58.6       | 58.6                   | sr11_external.json              |
| PASS        | S11.147 | SR 11     | ChEMBL approval rate, >=6 TAs without a PAV (%)                 | 28.0       | 28.0                   | sr11_external.json              |
| PASS        | S11.148 | SR 11     | ChEMBL approval rate, >=6 TAs with a PAV (%)                    | 29.9       | 29.9                   | sr11_external.json              |
| PASS        | S11.149 | SR 11     | interaction OR in Pharmaprojects                                | 6.39       | 6.39                   | sr11_external.json              |
| PASS        | S11.15  | SR 11     | approved pairs behind it                                        | 18         | 18                     | sr11_criterion.json             |
| PASS        | S11.150 | SR 11     | its CI low                                                      | 2.17       | 2.17                   | sr11_external.json              |
| PASS        | S11.151 | SR 11     | its CI high                                                     | 18.79      | 18.79                  | sr11_external.json              |
| PASS        | S11.152 | SR 11     | its likelihood-ratio P                                          | 0.0005     | 0.0004951567592193967  | sr11_external.json              |
| PASS        | S11.153 | SR 11     | its permutation P                                               | 0.001      | 0.000999900009999      | sr11_external.json              |
| PASS        | S11.154 | SR 11     | supported pairs in the Pharmaprojects model                     | 435        | 435                    | sr11_external.json              |
| PASS        | S11.155 | SR 11     | approved pairs among them                                       | 80         | 80                     | sr11_external.json              |
| PASS        | S11.156 | SR 11     | Pharmaprojects approval rate, 2-5 TAs without a PAV (%)         | 12.7       | 12.7                   | sr11_external.json              |
| PASS        | S11.157 | SR 11     | Pharmaprojects approval rate, 2-5 TAs with a PAV (%)            | 38.3       | 38.3                   | sr11_external.json              |
| PASS        | S11.158 | SR 11     | Pharmaprojects approval rate, >=6 TAs without a PAV (%)         | 18.3       | 18.3                   | sr11_external.json              |
| PASS        | S11.159 | SR 11     | Pharmaprojects approval rate, >=6 TAs with a PAV (%)            | 13.0       | 13.0                   | sr11_external.json              |
| PASS        | S11.16  | SR 11     | OR for the 2-4 window                                           | 10.35      | 10.35                  | sr11_criterion.json             |
| PASS        | S11.17  | SR 11     | approved pairs behind it                                        | 40         | 40                     | sr11_criterion.json             |
| PASS        | S11.18  | SR 11     | OR for the 1-5 window                                           | 8.99       | 8.99                   | sr11_criterion.json             |
| PASS        | S11.19  | SR 11     | OR for the 3-5 window                                           | 9.29       | 9.29                   | sr11_criterion.json             |
| PASS        | S11.20  | SR 11     | OR for the 2-6 window                                           | 9.47       | 9.47                   | sr11_criterion.json             |
| PASS        | S11.21  | SR 11     | decomposition, all GWAS support OR                              | 3.62       | 3.62                   | sr11_criterion.json             |
| PASS        | S11.22  | SR 11     | decomposition, all GWAS support RS                              | 2.76       | 2.76                   | sr11_criterion.json             |
| PASS        | S11.23  | SR 11     | decomposition, all GWAS approved pairs                          | 242        | 242                    | sr11_criterion.json             |
| PASS        | S11.24  | SR 11     | decomposition, PAV any TA OR                                    | 5.89       | 5.89                   | sr11_criterion.json             |
| PASS        | S11.25  | SR 11     | decomposition, PAV any TA RS                                    | 3.71       | 3.71                   | sr11_criterion.json             |
| PASS        | S11.26  | SR 11     | decomposition, PAV any TA approved pairs                        | 72         | 72                     | sr11_criterion.json             |
| PASS        | S11.27  | SR 11     | decomposition, any support 2-5 TAs OR                           | 3.95       | 3.95                   | sr11_criterion.json             |
| PASS        | S11.28  | SR 11     | decomposition, any support 2-5 TAs RS                           | 2.92       | 2.92                   | sr11_criterion.json             |
| PASS        | S11.29  | SR 11     | decomposition, any support 2-5 TAs approved pairs               | 139        | 139                    | sr11_criterion.json             |
| PASS        | S11.30  | SR 11     | decomposition, any support >=2 TAs OR                           | 3.54       | 3.54                   | sr11_criterion.json             |
| PASS        | S11.31  | SR 11     | decomposition, any support >=2 TAs RS                           | 2.72       | 2.72                   | sr11_criterion.json             |
| PASS        | S11.32  | SR 11     | decomposition, any support >=2 TAs approved pairs               | 220        | 220                    | sr11_criterion.json             |
| PASS        | S11.33  | SR 11     | decomposition, criterion OR                                     | 10.29      | 10.29                  | sr11_criterion.json             |
| PASS        | S11.34  | SR 11     | decomposition, criterion RS                                     | 4.84       | 4.84                   | sr11_criterion.json             |
| PASS        | S11.35  | SR 11     | decomposition, criterion approved pairs                         | 51         | 51                     | sr11_criterion.json             |
| PASS        | S11.36  | SR 11     | PAV multiplier over the baseline                                | 1.63       | 1.63                   | sr11_criterion.json             |
| PASS        | S11.37  | SR 11     | therapeutic-area window multiplier                              | 1.09       | 1.09                   | sr11_criterion.json             |
| PASS        | S11.38  | SR 11     | OR if the two components acted independently                    | 6.4        | 6.4                    | sr11_criterion.json             |
| PASS        | S11.39  | SR 11     | median held-out OR across 200 splits                            | 10.32      | 10.32                  | sr11_criterion.json             |
| PASS        | S11.40  | SR 11     | held-out spread, lower                                          | 6.62       | 6.62                   | sr11_criterion.json             |
| PASS        | S11.41  | SR 11     | held-out spread, upper                                          | 16.75      | 16.75                  | sr11_criterion.json             |
| PASS        | S11.42  | SR 11     | in-sample over held-out ratio                                   | 0.97       | 0.97                   | sr11_criterion.json             |
| PASS        | S11.43  | SR 11     | that ratio, Monte Carlo interval low                            | 0.91       | 0.91                   | sr11_criterion.json             |
| PASS        | S11.44  | SR 11     | that ratio, Monte Carlo interval high                           | 1.04       | 1.04                   | sr11_criterion.json             |
| PASS        | S11.45  | SR 11     | splits whose held-out CI excludes the baseline (%)              | 95.5       | 95.5                   | sr11_criterion.json             |
| PASS        | S11.46  | SR 11     | halves where PAV enrichment exceeds non-PAV (%)                 | 99.5       | 100.0                  | sr11_criterion.json             |
| PASS        | S11.47  | SR 11     | halves where that difference reaches P < 0.05 (%)               | 79.5       | 79.0                   | sr11_criterion.json             |
| PASS        | S11.48  | SR 11     | halves where the quadratic test reaches P < 0.05 (%)            | 100        | 99.5                   | sr11_criterion.json             |
| PASS        | S11.49  | SR 11     | halves where it reaches P < 1e-4 (%)                            | 92.5       | 89.0                   | sr11_criterion.json             |
| PASS        | S11.50  | SR 11     | optimism factor from searching for the window                   | 1.19       | 1.19                   | sr11_criterion.json             |
| PASS        | S11.51  | SR 11     | optimism-corrected OR                                           | 8.63       | 8.63                   | sr11_criterion.json             |
| PASS        | S11.52  | SR 11     | corrected CI low                                                | 5.63       | 5.63                   | sr11_criterion.json             |
| PASS        | S11.53  | SR 11     | corrected CI high                                               | 13.24      | 13.24                  | sr11_criterion.json             |
| PASS        | S11.54  | SR 11     | optimism-corrected relative success                             | 4.50       | 4.5                    | sr11_criterion.json             |
| PASS        | S11.55  | SR 11     | Pharmaprojects target-indication pairs                          | 7390       | 7390                   | sr11_external.json              |
| PASS        | S11.56  | SR 11     | of those, maximum Phase I                                       | 2274       | 2274                   | sr11_external.json              |
| PASS        | S11.57  | SR 11     | maximum Phase II                                                | 3303       | 3303                   | sr11_external.json              |
| PASS        | S11.58  | SR 11     | maximum Phase III                                               | 900        | 900                    | sr11_external.json              |
| PASS        | S11.59  | SR 11     | launched                                                        | 913        | 913                    | sr11_external.json              |
| PASS        | S11.60  | SR 11     | launched pairs also Phase IV in ChEMBL                          | 447        | 447                    | sr11_external.json              |
| PASS        | S11.61  | SR 11     | that as a share of launched pairs (%)                           | 49         | 49.0                   | sr11_external.json              |
| PASS        | S11.62  | SR 11     | our genetic support in Pharmaprojects, OR                       | 1.65       | 1.65                   | sr11_external.json              |
| PASS        | S11.63  | SR 11     | its CI low                                                      | 1.3        | 1.3                    | sr11_external.json              |
| PASS        | S11.64  | SR 11     | its CI high                                                     | 2.11       | 2.11                   | sr11_external.json              |
| PASS        | S11.65  | SR 11     | its P                                                           | 0.00011    | 0.0001124840258449383  | sr11_external.json              |
| PASS        | S11.66  | SR 11     | pairs where our support and theirs agree                        | 287        | 287                    | sr11_external.json              |
| PASS        | S11.67  | SR 11     | pairs our support covers                                        | 469        | 469                    | sr11_external.json              |
| PASS        | S11.68  | SR 11     | pairs their annotation covers                                   | 858        | 858                    | sr11_external.json              |
| PASS        | S11.69  | SR 11     | their own annotation, OR                                        | 2.32       | 2.32                   | sr11_external.json              |
| PASS        | S11.70  | SR 11     | its CI low                                                      | 1.94       | 1.94                   | sr11_external.json              |
| PASS        | S11.71  | SR 11     | its CI high                                                     | 2.78       | 2.78                   | sr11_external.json              |
| PASS        | S11.72  | SR 11     | its P                                                           | 1.8e-20    | 1.8382242483606243e-20 | sr11_external.json              |
| PASS        | S11.73  | SR 11     | its relative success                                            | 2.03       | 2.03                   | sr11_external.json              |
| PASS        | S11.74  | SR 11     | Pharmaprojects PAV support OR                                   | 2.34       | 2.34                   | sr11_external.json              |
| PASS        | S11.77  | SR 11     | Pharmaprojects non-PAV support OR                               | 1.4        | 1.4                    | sr11_external.json              |
| PASS        | S11.80  | SR 11     | P for the difference between them                               | 0.04       | 0.04                   | sr11_external.json              |
| PASS        | S11.81  | SR 11     | Pharmaprojects gPS <= 5 OR                                      | 1.92       | 1.92                   | sr11_external.json              |
| PASS        | S11.82  | SR 11     | Pharmaprojects gPS >= 10 OR                                     | 1.62       | 1.62                   | sr11_external.json              |
| PASS        | S11.83  | SR 11     | ratio between them                                              | 1.18       | 1.18                   | sr11_external.json              |
| PASS        | S11.84  | SR 11     | P for that difference                                           | 0.54       | 0.54                   | sr11_external.json              |
| PASS        | S11.85  | SR 11     | ChEMBL gPS <= 5 OR                                              | 4.8        | 4.8                    | sr11_external.json              |
| PASS        | S11.86  | SR 11     | ChEMBL gPS >= 10 OR                                             | 2.97       | 2.97                   | sr11_external.json              |
| PASS        | S11.87  | SR 11     | ratio between them                                              | 1.62       | 1.62                   | sr11_external.json              |
| PASS        | S11.88  | SR 11     | P for that difference                                           | 0.0077     | 0.0077                 | sr11_external.json              |
| PASS        | S11.89  | SR 11     | criterion applied to Pharmaprojects, OR                         | 4.5        | 4.5                    | sr11_external.json              |
| PASS        | S11.90  | SR 11     | its CI low                                                      | 2.66       | 2.66                   | sr11_external.json              |
| PASS        | S11.91  | SR 11     | its CI high                                                     | 7.6        | 7.6                    | sr11_external.json              |
| PASS        | S11.92  | SR 11     | its relative success                                            | 3.16       | 3.16                   | sr11_external.json              |
| PASS        | S11.93  | SR 11     | its P                                                           | 2.5e-07    | 2.5245428220915465e-07 | sr11_external.json              |
| PASS        | S11.94  | SR 11     | launched pairs meeting it                                       | 23         | 23                     | sr11_external.json              |
| PASS        | S11.95  | SR 11     | supported pairs meeting it                                      | 60         | 60                     | sr11_external.json              |
| PASS        | S11.96  | SR 11     | lift over the Pharmaprojects baseline                           | 2.72       | 2.72                   | sr11_external.json              |
| PASS        | S11.97  | SR 11     | lift over the ChEMBL baseline                                   | 2.84       | 2.84                   | sr11_external.json              |
| PASS        | S11.98  | SR 11     | quadratic LR in Pharmaprojects                                  | 0.47       | 0.47                   | sr11_external.json              |
| PASS        | S11.99  | SR 11     | its P                                                           | 0.49       | 0.49                   | sr11_external.json              |
| PASS        | S12.01  | SR 12     | disease studies modelled                                        | 5349       | 5349                   | sr12_discovery_regression.json  |
| PASS        | S12.02  | SR 12     | cohort clusters                                                 | 1525       | 1525                   | sr12_discovery_regression.json  |
| PASS        | S12.03  | SR 12     | variance over mean of novel genes                               | 30         | 21.0                   | sr12_discovery_regression.json  |
| PASS        | S12.04  | SR 12     | IRR, fully non-European vs fully European                       | 1.69       | 1.69                   | sr12_discovery_regression.json  |
| PASS        | S12.05  | SR 12     | its CI low                                                      | 1.3        | 1.3                    | sr12_discovery_regression.json  |
| PASS        | S12.06  | SR 12     | its CI high                                                     | 2.2        | 2.2                    | sr12_discovery_regression.json  |
| PASS        | S12.07  | SR 12     | its P                                                           | 0.0001     | 0.0001017778783449122  | sr12_discovery_regression.json  |
| PASS        | S12.08  | SR 12     | IRR per unit European fraction                                  | 0.59       | 0.59                   | sr12_discovery_regression.json  |
| PASS        | S12.09  | SR 12     | IRR for a tenfold larger study                                  | 3.67       | 3.67                   | sr12_discovery_regression.json  |
| PASS        | S12.10  | SR 12     | its CI low                                                      | 2.75       | 2.75                   | sr12_discovery_regression.json  |
| PASS        | S12.11  | SR 12     | its CI high                                                     | 4.89       | 4.89                   | sr12_discovery_regression.json  |
| PASS        | S12.12  | SR 12     | its P                                                           | 8e-19      | 7.719059835773669e-19  | sr12_discovery_regression.json  |
| PASS        | S12.13  | SR 12     | IRR per publication year                                        | 0.84       | 0.84                   | sr12_discovery_regression.json  |
| PASS        | S12.14  | SR 12     | its CI low                                                      | 0.79       | 0.79                   | sr12_discovery_regression.json  |
| PASS        | S12.15  | SR 12     | its CI high                                                     | 0.88       | 0.88                   | sr12_discovery_regression.json  |
| PASS        | S12.16  | SR 12     | its P                                                           | 4e-11      | 4.299803303664462e-11  | sr12_discovery_regression.json  |
| PASS        | S12.17  | SR 12     | IRR, non-EUR class vs EUR                                       | 1.67       | 1.67                   | sr12_discovery_regression.json  |
| PASS        | S12.18  | SR 12     | its CI low                                                      | 1.3        | 1.3                    | sr12_discovery_regression.json  |
| PASS        | S12.19  | SR 12     | its CI high                                                     | 2.14       | 2.14                   | sr12_discovery_regression.json  |
| PASS        | S12.20  | SR 12     | its P                                                           | 0.0001     | 5.176992464300313e-05  | sr12_discovery_regression.json  |
| PASS        | S12.21  | SR 12     | IRR, mixed class vs EUR                                         | 0.97       | 0.97                   | sr12_discovery_regression.json  |
| PASS        | S12.22  | SR 12     | its CI low                                                      | 0.8        | 0.8                    | sr12_discovery_regression.json  |
| PASS        | S12.23  | SR 12     | its CI high                                                     | 1.18       | 1.18                   | sr12_discovery_regression.json  |
| PASS        | S12.24  | SR 12     | its P                                                           | 0.78       | 0.78                   | sr12_discovery_regression.json  |
| PASS        | S12.25  | SR 12     | measurements, continuous IRR                                    | 3.37       | 3.37                   | sr12_discovery_regression.json  |
| PASS        | S12.26  | SR 12     | measurements, non-EUR class IRR                                 | 3.84       | 3.84                   | sr12_discovery_regression.json  |
| PASS        | S12.27  | SR 12     | its CI low                                                      | 1.99       | 1.99                   | sr12_discovery_regression.json  |
| PASS        | S12.28  | SR 12     | its CI high                                                     | 7.43       | 7.43                   | sr12_discovery_regression.json  |
| PASS        | S12.29  | SR 12     | measurements, mixed class IRR                                   | 0.62       | 0.62                   | sr12_discovery_regression.json  |
| PASS        | S12.30  | SR 12     | its CI low                                                      | 0.31       | 0.31                   | sr12_discovery_regression.json  |
| PASS        | S12.31  | SR 12     | its CI high                                                     | 1.27       | 1.27                   | sr12_discovery_regression.json  |
| PASS        | S12.32  | SR 12     | its P                                                           | 0.19       | 0.19                   | sr12_discovery_regression.json  |
| PASS        | S12.33  | SR 12     | measurements, IRR for a tenfold larger study                    | 5.95       | 5.95                   | sr12_discovery_regression.json  |
| PASS        | S12.34  | SR 12     | its CI low                                                      | 3.75       | 3.75                   | sr12_discovery_regression.json  |
| PASS        | S12.35  | SR 12     | its CI high                                                     | 9.43       | 9.43                   | sr12_discovery_regression.json  |
| PASS        | S12.36  | SR 12     | IRR with year as a factor                                       | 2.01       | 2.01                   | sr12_discovery_regression.json  |
| PASS        | S12.37  | SR 12     | IRR with year omitted                                           | 0.9        | 0.9                    | sr12_discovery_regression.json  |
| PASS        | S12.38  | SR 12     | its P                                                           | 0.29       | 0.29                   | sr12_discovery_regression.json  |
| PASS        | S12.39  | SR 12     | European share of studies up to 2017 (%)                        | 76.9       | 76.9                   | sr12_discovery_regression.json  |
| PASS        | S12.40  | SR 12     | European share from 2018 (%)                                    | 63.8       | 64.5                   | sr12_discovery_regression.json  |
| PASS        | S12.41  | SR 12     | IRR with all prioritised genes as the outcome                   | 1.51       | 1.51                   | sr12_discovery_regression.json  |
| PASS        | S12.42  | SR 12     | IRR with credible sets as the outcome                           | 1.55       | 1.55                   | sr12_discovery_regression.json  |
| PASS        | S12.43  | SR 12     | novel genes, top size quintile, under 10% European              | 8.07       | 8.07                   | sr12_discovery_regression.json  |
| PASS        | S12.44  | SR 12     | novel genes, top size quintile, over 90% European               | 4.42       | 4.42                   | sr12_discovery_regression.json  |
| PASS        | S12.45  | SR 12     | lowest IRR across leave-one-cohort-out refits                   | 1.46       | 1.46                   | sr12_discovery_regression.json  |
| PASS        | S12.46  | SR 12     | highest IRR across those refits                                 | 1.72       | 1.72                   | sr12_discovery_regression.json  |
| PASS        | S12.47  | SR 12     | IRR excluding FinnGen                                           | 1.46       | 1.46                   | sr12_discovery_regression.json  |
| PASS        | S12.48  | SR 12     | its CI low                                                      | 0.99       | 0.99                   | sr12_discovery_regression.json  |
| PASS        | S12.49  | SR 12     | its CI high                                                     | 2.14       | 2.14                   | sr12_discovery_regression.json  |
| PASS        | S12.50  | SR 12     | its P                                                           | 0.055      | 0.055                  | sr12_discovery_regression.json  |
| PASS        | S14.01  | SR 14     | traits in the genetic correlation matrix                        | 1114       | 1114                   | sr14_genetic_correlation.json   |
| PASS        | S14.02  | SR 14     | of those, diseases                                              | 551        | 551                    | sr14_genetic_correlation.json   |
| PASS        | S14.03  | SR 14     | of those, measurements                                          | 563        | 563                    | sr14_genetic_correlation.json   |
| PASS        | S14.04  | SR 14     | off-diagonal entries with a measured correlation (%)            | 99.84      | 99.84                  | sr14_genetic_correlation.json   |
| PASS        | S14.05  | SR 14     | disease terms present in the matrix                             | 471        | 471                    | sr14_genetic_correlation.json   |
| PASS        | S14.06  | SR 14     | those as a share of disease terms (%)                           | 33.8       | 33.8                   | sr14_genetic_correlation.json   |
| PASS        | S14.07  | SR 14     | measurement terms present in the matrix                         | 507        | 507                    | sr14_genetic_correlation.json   |
| PASS        | S14.08  | SR 14     | those as a share of measurement terms (%)                       | 14.9       | 14.9                   | sr14_genetic_correlation.json   |
| PASS        | S14.09  | SR 14     | gene-disease associations covered                               | 27006      | 27006                  | sr14_genetic_correlation.json   |
| PASS        | S14.10  | SR 14     | those as a share of gene-disease associations (%)               | 73.3       | 73.3                   | sr14_genetic_correlation.json   |
| PASS        | S14.11  | SR 14     | gene-measurement associations covered                           | 86522      | 86522                  | sr14_genetic_correlation.json   |
| PASS        | S14.12  | SR 14     | those as a share of gene-measurement associations (%)           | 75.2       | 75.2                   | sr14_genetic_correlation.json   |
| PASS        | S14.13  | SR 14     | disease traits carrying a therapeutic area                      | 400        | 400                    | sr14_genetic_correlation.json   |
| PASS        | S14.14  | SR 14     | disease traits under no therapeutic-area root                   | 151        | 151                    | sr14_genetic_correlation.json   |
| PASS        | S14.15  | SR 14     | therapeutic areas represented                                   | 21         | 21                     | sr14_genetic_correlation.json   |
| PASS        | S14.16  | SR 14     | within-area disease pairs                                       | 5844       | 5844                   | sr14_genetic_correlation.json   |
| PASS        | S14.17  | SR 14     | between-area disease pairs                                      | 73956      | 73956                  | sr14_genetic_correlation.json   |
| PASS        | S14.18  | SR 14     | mean absolute correlation within an area                        | 0.401      | 0.401                  | sr14_genetic_correlation.json   |
| PASS        | S14.19  | SR 14     | mean absolute correlation between areas                         | 0.317      | 0.317                  | sr14_genetic_correlation.json   |
| PASS        | S14.20  | SR 14     | difference between them                                         | 0.084      | 0.084                  | sr14_genetic_correlation.json   |
| PASS        | S14.21  | SR 14     | fold difference in pairs above 0.5                              | 1.48       | 1.48                   | sr14_genetic_correlation.json   |
| PASS        | S14.22  | SR 14     | superiority of a within-area pair                               | 0.574      | 0.574                  | sr14_genetic_correlation.json   |
| PASS        | S14.23  | SR 14     | nested pairs removed                                            | 1007       | 1007                   | sr14_genetic_correlation.json   |
| PASS        | S14.24  | SR 14     | within-area mean after removing them                            | 0.390      | 0.39                   | sr14_genetic_correlation.json   |
| PASS        | S14.25  | SR 14     | between-area mean after removing them                           | 0.316      | 0.316                  | sr14_genetic_correlation.json   |
| PASS        | S14.26  | SR 14     | fold difference after removing them                             | 1.42       | 1.42                   | sr14_genetic_correlation.json   |
| PASS        | S14.27  | SR 14     | superiority after removing them                                 | 0.563      | 0.563                  | sr14_genetic_correlation.json   |
| PASS        | S14.28  | SR 14     | within-area pairs after removing them                           | 5103       | 5103                   | sr14_genetic_correlation.json   |
| PASS        | S14.29  | SR 14     | between-area pairs after removing them                          | 73690      | 73690                  | sr14_genetic_correlation.json   |
| PASS        | S14.30  | SR 14     | between-area pairs above 0.5 (%)                                | 22         | 22.0                   | sr14_genetic_correlation.json   |
| PASS        | S2.03   | SR 2      | qualified CSs colocalising with any molQTL                      | 330584     | 330584                 | sr02_colocalisation.json        |
| PASS        | S2.04   | SR 2      | those as a share of qualified CSs (%)                           | 63         | 63.0                   | sr02_colocalisation.json        |
| PASS        | S2.05   | SR 2      | excluding trans-pQTL colocalisations                            | 302264     | 302264                 | sr02_colocalisation.json        |
| PASS        | S2.06   | SR 2      | those as a share of qualified CSs (%)                           | 58         | 58.0                   | sr02_colocalisation.json        |
| PASS        | S2.07   | SR 2      | with a protein-coding gene molQTL colocalisation                | 285229     | 285229                 | sr02_colocalisation.json        |
| PASS        | S2.08   | SR 2      | those as a share of qualified CSs (%)                           | 55         | 55.0                   | sr02_colocalisation.json        |
| PASS        | S2.09   | SR 2      | unique protein-coding genes in those colocalisations            | 14026      | 14026                  | sr02_colocalisation.json        |
| PASS        | S3.01   | SR 3      | qualified CSs with a protein-coding gene in the feature matrix  | 513568     | 513568                 | sr03_l2g.json                   |
| PASS        | S3.02   | SR 3      | CS-gene pairs in the feature matrix                             | 7066749    | 7066749                | sr03_l2g.json                   |
| PASS        | S3.03   | SR 3      | mean genes assigned per CS                                      | 13.76      | 13.76                  | sr03_l2g.json                   |
| PASS        | S3.04   | SR 3      | median genes assigned per CS                                    | 10         | 10                     | sr03_l2g.json                   |
| PASS        | S3.05   | SR 3      | positive gene-CS pairs in the gold standard                     | 8520       | 8520                   | sr03_l2g.json                   |
| PASS        | S3.06   | SR 3      | unique positive gene-EFO pairs                                  | 1377       | 1377                   | sr03_l2g.json                   |
| PASS        | S3.07   | SR 3      | unique positive genes                                           | 390        | 390                    | sr03_l2g.json                   |
| PASS        | S3.08   | SR 3      | training set positives                                          | 7386       | 7386                   | sr03_l2g.json                   |
| PASS        | S3.09   | SR 3      | training set negatives                                          | 106973     | 106973                 | sr03_l2g.json                   |
| PASS        | S3.10   | SR 3      | test set positives                                              | 1134       | 1134                   | sr03_l2g.json                   |
| PASS        | S3.11   | SR 3      | test set negatives                                              | 17477      | 17477                  | sr03_l2g.json                   |
| PASS        | S3.12   | SR 3      | negatives per positive                                          | 14.6       | 14.6                   | sr03_l2g.json                   |
| PASS        | S3.13   | SR 3      | positive gene-EFO pairs whose gene is nearest the TSS           | 773        | 773                    | sr03_l2g.json                   |
| PASS        | S3.14   | SR 3      | those as a share of positives (%)                               | 56.1       | 56.1                   | sr03_l2g.json                   |
| PASS        | S3.17   | SR 3      | precision at L2G >= 0.5                                         | 0.885      | 0.885                  | sr03_l2g.json                   |
| PASS        | S3.21   | SR 3      | held-out false positives                                        | 95         | 95                     | sr03_l2g.json                   |
| PASS        | S3.22   | SR 3      | held-out false negatives                                        | 402        | 402                    | sr03_l2g.json                   |
| PASS        | S3.23   | SR 3      | held-out true positives                                         | 732        | 732                    | sr03_l2g.json                   |
| PASS        | S3.24   | SR 3      | CSs with exactly one gene at L2G >= 0.5                         | 309989     | 309989                 | sr03_l2g.json                   |
| PASS        | S3.25   | SR 3      | those as a share of qualified CSs (%)                           | 59.5       | 59.5                   | sr03_l2g.json                   |
| PASS        | S3.30   | SR 3      | prioritisations with an eQTL colocalisation                     | 191871     | 191871                 | sr03_l2g.json                   |
| PASS        | S3.31   | SR 3      | those as a share of prioritisations (%)                         | 36.7       | 36.7                   | sr03_l2g.json                   |
| PASS        | S3.32   | SR 3      | prioritisations with a pQTL colocalisation                      | 30030      | 30030                  | sr03_l2g.json                   |
| PASS        | S3.33   | SR 3      | those as a share of prioritisations (%)                         | 5.7        | 5.7                    | sr03_l2g.json                   |
| PASS        | S3.34   | SR 3      | prioritisations with a protein-altering variant                 | 63327      | 63327                  | sr03_l2g.json                   |
| PASS        | S3.36   | SR 3      | prioritisations nearest to a TSS                                | 424781     | 424781                 | sr03_l2g.json                   |
| PASS        | S3.37   | SR 3      | those as a share of prioritisations (%)                         | 81.2       | 81.2                   | sr03_l2g.json                   |
| PASS        | S3.38   | SR 3      | prioritisations with no PAV, eQTL or pQTL                       | 241404     | 241404                 | sr03_l2g.json                   |
| PASS        | S3.39   | SR 3      | those as a share of prioritisations (%)                         | 46.1       | 46.1                   | sr03_l2g.json                   |
| PASS        | S4.01   | SR 4      | L2G > 0.5, sensitivity, held-out set                            | 0.646      | 0.646                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.02   | SR 4      | L2G > 0.5, specificity, held-out set                            | 0.995      | 0.995                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.03   | SR 4      | L2G > 0.5, PPV, held-out set                                    | 0.885      | 0.885                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.04   | SR 4      | L2G > 0.5, FDR, held-out set                                    | 0.115      | 0.115                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.05   | SR 4      | L2G > 0.5, sensitivity, training and test sets                  | 0.736      | 0.736                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.06   | SR 4      | L2G > 0.5, specificity, training and test sets                  | 0.995      | 0.995                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.07   | SR 4      | L2G > 0.5, PPV, training and test sets                          | 0.904      | 0.904                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.08   | SR 4      | L2G > 0.5, FDR, training and test sets                          | 0.096      | 0.096                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.09   | SR 4      | L2G > 0.005, sensitivity, held-out set                          | 0.839      | 0.839                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.10   | SR 4      | L2G > 0.005, specificity, held-out set                          | 0.932      | 0.932                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.11   | SR 4      | L2G > 0.005, PPV, held-out set                                  | 0.443      | 0.443                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.12   | SR 4      | L2G > 0.005, FDR, held-out set                                  | 0.557      | 0.557                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.13   | SR 4      | L2G > 0.005, sensitivity, training and test sets                | 0.88       | 0.88                   | sr04_l2g_vs_naive.json          |
| PASS        | S4.14   | SR 4      | L2G > 0.005, specificity, training and test sets                | 0.948      | 0.948                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.15   | SR 4      | L2G > 0.005, PPV, training and test sets                        | 0.538      | 0.538                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.16   | SR 4      | L2G > 0.005, FDR, training and test sets                        | 0.462      | 0.462                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.17   | SR 4      | L2G > 0.8, sensitivity, held-out set                            | 0.339      | 0.339                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.18   | SR 4      | L2G > 0.8, specificity, held-out set                            | 0.998      | 0.998                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.19   | SR 4      | L2G > 0.8, PPV, held-out set                                    | 0.923      | 0.923                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.20   | SR 4      | L2G > 0.8, FDR, held-out set                                    | 0.077      | 0.077                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.21   | SR 4      | L2G > 0.8, sensitivity, training and test sets                  | 0.528      | 0.528                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.22   | SR 4      | L2G > 0.8, specificity, training and test sets                  | 0.999      | 0.999                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.23   | SR 4      | L2G > 0.8, PPV, training and test sets                          | 0.965      | 0.965                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.24   | SR 4      | L2G > 0.8, FDR, training and test sets                          | 0.035      | 0.035                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.25   | SR 4      | eQTL coloc, sensitivity, held-out set                           | 0.218      | 0.218                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.26   | SR 4      | eQTL coloc, specificity, held-out set                           | 0.974      | 0.974                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.27   | SR 4      | eQTL coloc, PPV, held-out set                                   | 0.349      | 0.349                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.28   | SR 4      | eQTL coloc, FDR, held-out set                                   | 0.651      | 0.651                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.29   | SR 4      | eQTL coloc, sensitivity, training and test sets                 | 0.348      | 0.348                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.30   | SR 4      | eQTL coloc, specificity, training and test sets                 | 0.96       | 0.96                   | sr04_l2g_vs_naive.json          |
| PASS        | S4.31   | SR 4      | eQTL coloc, PPV, training and test sets                         | 0.375      | 0.375                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.32   | SR 4      | eQTL coloc, FDR, training and test sets                         | 0.625      | 0.625                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.33   | SR 4      | pQTL coloc, sensitivity, held-out set                           | 0.184      | 0.184                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.34   | SR 4      | pQTL coloc, specificity, held-out set                           | 0.992      | 0.992                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.35   | SR 4      | pQTL coloc, PPV, held-out set                                   | 0.613      | 0.613                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.36   | SR 4      | pQTL coloc, FDR, held-out set                                   | 0.387      | 0.387                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.37   | SR 4      | pQTL coloc, sensitivity, training and test sets                 | 0.141      | 0.141                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.38   | SR 4      | pQTL coloc, specificity, training and test sets                 | 0.997      | 0.997                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.39   | SR 4      | pQTL coloc, PPV, training and test sets                         | 0.782      | 0.782                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.40   | SR 4      | pQTL coloc, FDR, training and test sets                         | 0.218      | 0.218                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.41   | SR 4      | PAV, sensitivity, held-out set                                  | 0.248      | 0.248                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.42   | SR 4      | PAV, specificity, held-out set                                  | 0.996      | 0.996                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.43   | SR 4      | PAV, PPV, held-out set                                          | 0.807      | 0.807                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.44   | SR 4      | PAV, FDR, held-out set                                          | 0.193      | 0.193                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.45   | SR 4      | PAV, sensitivity, training and test sets                        | 0.207      | 0.207                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.46   | SR 4      | PAV, specificity, training and test sets                        | 0.997      | 0.997                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.47   | SR 4      | PAV, PPV, training and test sets                                | 0.804      | 0.804                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.48   | SR 4      | PAV, FDR, training and test sets                                | 0.196      | 0.196                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.49   | SR 4      | Nearest, sensitivity, held-out set                              | 0.702      | 0.702                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.50   | SR 4      | Nearest, specificity, held-out set                              | 0.983      | 0.983                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.51   | SR 4      | Nearest, PPV, held-out set                                      | 0.73       | 0.73                   | sr04_l2g_vs_naive.json          |
| PASS        | S4.52   | SR 4      | Nearest, FDR, held-out set                                      | 0.27       | 0.27                   | sr04_l2g_vs_naive.json          |
| PASS        | S4.53   | SR 4      | Nearest, sensitivity, training and test sets                    | 0.644      | 0.644                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.54   | SR 4      | Nearest, specificity, training and test sets                    | 0.982      | 0.982                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.55   | SR 4      | Nearest, PPV, training and test sets                            | 0.708      | 0.708                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.56   | SR 4      | Nearest, FDR, training and test sets                            | 0.292      | 0.292                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.57   | SR 4      | Combined, sensitivity, held-out set                             | 0.776      | 0.776                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.58   | SR 4      | Combined, specificity, held-out set                             | 0.986      | 0.986                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.59   | SR 4      | Combined, PPV, held-out set                                     | 0.788      | 0.788                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.60   | SR 4      | Combined, FDR, held-out set                                     | 0.212      | 0.212                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.61   | SR 4      | Combined, sensitivity, training and test sets                   | 0.798      | 0.798                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.62   | SR 4      | Combined, specificity, training and test sets                   | 0.989      | 0.989                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.63   | SR 4      | Combined, PPV, training and test sets                           | 0.831      | 0.831                  | sr04_l2g_vs_naive.json          |
| PASS        | S4.64   | SR 4      | Combined, FDR, training and test sets                           | 0.169      | 0.169                  | sr04_l2g_vs_naive.json          |
| PASS        | S5.01   | SR 5      | disease-associated GWAS regions                                 | 24558      | 24558                  | sr05_secondary_signals.json     |
| PASS        | S5.02   | SR 5      | regions with both a primary and a secondary CS                  | 6354       | 6354                   | sr05_secondary_signals.json     |
| PASS        | S5.03   | SR 5      | primary credible sets in those regions                          | 8780       | 8780                   | sr05_secondary_signals.json     |
| PASS        | S5.04   | SR 5      | secondary credible sets in those regions                        | 11439      | 11439                  | sr05_secondary_signals.json     |
| PASS        | S5.05   | SR 5      | regions whose primary CS carries no evidence                    | 2862       | 2862                   | sr05_secondary_signals.json     |
| PASS        | S5.07   | SR 5      | primary CSs with an eQTL colocalisation (%)                     | 48.5       | 48.5                   | sr05_secondary_signals.json     |
| PASS        | S5.08   | SR 5      | secondary CSs with an eQTL colocalisation (%)                   | 42.4       | 42.4                   | sr05_secondary_signals.json     |
| PASS        | S5.09   | SR 5      | primary CSs with a pQTL colocalisation (%)                      | 9.8        | 9.8                    | sr05_secondary_signals.json     |
| PASS        | S5.10   | SR 5      | secondary CSs with a pQTL colocalisation (%)                    | 7.2        | 7.2                    | sr05_secondary_signals.json     |
| PASS        | S5.11   | SR 5      | primary CSs with a protein-altering variant (%)                 | 17.5       | 17.5                   | sr05_secondary_signals.json     |
| PASS        | S5.12   | SR 5      | secondary CSs with a protein-altering variant (%)               | 14.6       | 14.6                   | sr05_secondary_signals.json     |
| PASS        | S6.01   | SR 6      | disease-associated variants in the effect matrix                | 40706      | 40706                  | sr06_variant_pleiotropy.json    |
| PASS        | S6.02   | SR 6      | diseases in the effect matrix                                   | 1403       | 1403                   | sr06_variant_pleiotropy.json    |
| PASS        | S6.03   | SR 6      | pleiotropic lead variants                                       | 9828       | 9828                   | sr06_variant_pleiotropy.json    |
| PASS        | S6.04   | SR 6      | those as a share of all variants (%)                            | 24         | 24.0                   | sr06_variant_pleiotropy.json    |
| PASS        | S6.05   | SR 6      | mean diseases per variant                                       | 1.48       | 1.48                   | sr06_variant_pleiotropy.json    |
| PASS        | S6.06   | SR 6      | maximum diseases per variant                                    | 85         | 85                     | sr06_variant_pleiotropy.json    |
| PASS        | S6.08   | SR 6      | those as a share of pleiotropic variants (%)                    | 18         | 18.0                   | sr06_variant_pleiotropy.json    |
| PASS        | S6.09   | SR 6      | highly pleiotropic variants with a fitted mixture               | 118        | 118                    | sr06_variant_pleiotropy.json    |
| PASS        | S6.10   | SR 6      | those favouring two components                                  | 100        | 100                    | sr06_variant_pleiotropy.json    |
| PASS        | S6.11   | SR 6      | those as a share of fitted variants (%)                         | 85         | 85.0                   | sr06_variant_pleiotropy.json    |
| PASS        | S6.12   | SR 6      | mean ratio between component means                              | 14.5       | 14.5                   | sr06_variant_pleiotropy.json    |
| PASS        | S6.13   | SR 6      | median ratio between component means                            | 7.7        | 7.7                    | sr06_variant_pleiotropy.json    |
| PASS        | S6.14   | SR 6      | share of associations in the large-effect component (%)         | 22         | 22.0                   | sr06_variant_pleiotropy.json    |
| PASS        | S6.15   | SR 6      | colocalisation clusters in Supplementary Figure 5               | 20041      | 20041                  | sr06_variant_pleiotropy.json    |
| PASS        | S6.17   | SR 6      | clusters at one disease and one therapeutic area                | 13424      | 13424                  | sr06_variant_pleiotropy.json    |
| PASS        | S7.01   | SR 7      | disease-associated genes in the models                          | 8285       | 8285                   | sr07_gps_discordance.json       |
| PASS        | S7.02   | SR 7      | gPS beta with mean discordance in the model                     | 0.29       | 0.29                   | sr07_gps_discordance.json       |
| PASS        | S7.03   | SR 7      | gPS P with mean discordance in the model                        | 5.4e-11    | 4.933984640282021e-11  | sr07_gps_discordance.json       |
| PASS        | S8.01   | SR 8      | OR for genetic support, no covariates                           | 3.62       | 3.62                   | sr08_enrichment_bias.json       |
| PASS        | S8.02   | SR 8      | OR adjusting for maximum sample size                            | 3.44       | 3.44                   | sr08_enrichment_bias.json       |
| PASS        | S8.04   | SR 8      | OR with both sample size and therapeutic area                   | 3.14       | 3.14                   | sr08_enrichment_bias.json       |
| PASS        | S9.01   | SR 9      | TA quadratic vs baseline LR                                     | 80.54      | 80.54                  | sr09_nonlinearity.json          |
| PASS        | S9.02   | SR 9      | TA quadratic vs baseline P                                      | 3.24e-18   | 3.2440423346077276e-18 | sr09_nonlinearity.json          |
| PASS        | S9.03   | SR 9      | TA quadratic vs linear LR                                       | 64.90      | 64.9                   | sr09_nonlinearity.json          |
| PASS        | S9.04   | SR 9      | TA quadratic vs linear P                                        | 7.89e-16   | 7.890354554856575e-16  | sr09_nonlinearity.json          |
| PASS        | S9.05   | SR 9      | gPS quadratic vs baseline LR                                    | 66.38      | 66.38                  | sr09_nonlinearity.json          |
| PASS        | S9.06   | SR 9      | gPS quadratic vs baseline P                                     | 3.85e-15   | 3.847333211503013e-15  | sr09_nonlinearity.json          |
| PASS        | S9.07   | SR 9      | gPS quadratic vs linear LR                                      | 54.17      | 54.17                  | sr09_nonlinearity.json          |
| PASS        | S9.08   | SR 9      | gPS quadratic vs linear P                                       | 1.84e-13   | 1.83697388968918e-13   | sr09_nonlinearity.json          |
| PASS        | S9.09   | SR 9      | TA linear log coefficient                                       | 0.568      | 0.568                  | sr09_nonlinearity.json          |
| PASS        | S9.10   | SR 9      | TA linear bootstrap mean                                        | 0.568      | 0.567                  | sr09_nonlinearity.json          |
| PASS        | S9.11   | SR 9      | TA linear bootstrap SD                                          | 0.064      | 0.064                  | sr09_nonlinearity.json          |
| PASS        | S9.12   | SR 9      | TA linear bootstrap CI low                                      | 0.441      | 0.444                  | sr09_nonlinearity.json          |
| PASS        | S9.13   | SR 9      | TA linear bootstrap CI high                                     | 0.692      | 0.69                   | sr09_nonlinearity.json          |
| PASS        | S9.14   | SR 9      | TA quadratic log coefficient                                    | -0.267     | -0.267                 | sr09_nonlinearity.json          |
| PASS        | S9.15   | SR 9      | TA quadratic bootstrap mean                                     | -0.266     | -0.266                 | sr09_nonlinearity.json          |
| PASS        | S9.16   | SR 9      | TA quadratic bootstrap SD                                       | 0.034      | 0.034                  | sr09_nonlinearity.json          |
| PASS        | S9.17   | SR 9      | TA quadratic bootstrap CI low                                   | -0.332     | -0.333                 | sr09_nonlinearity.json          |
| PASS        | S9.18   | SR 9      | TA quadratic bootstrap CI high                                  | -0.198     | -0.199                 | sr09_nonlinearity.json          |
| PASS        | S9.19   | SR 9      | gPS linear log coefficient                                      | 0.371      | 0.371                  | sr09_nonlinearity.json          |
| PASS        | S9.20   | SR 9      | gPS linear bootstrap mean                                       | 0.370      | 0.37                   | sr09_nonlinearity.json          |
| PASS        | S9.21   | SR 9      | gPS linear bootstrap SD                                         | 0.047      | 0.047                  | sr09_nonlinearity.json          |
| PASS        | S9.22   | SR 9      | gPS linear bootstrap CI low                                     | 0.271      | 0.279                  | sr09_nonlinearity.json          |
| PASS        | S9.23   | SR 9      | gPS linear bootstrap CI high                                    | 0.459      | 0.456                  | sr09_nonlinearity.json          |
| PASS        | S9.24   | SR 9      | gPS quadratic log coefficient                                   | -0.120     | -0.12                  | sr09_nonlinearity.json          |
| PASS        | S9.25   | SR 9      | gPS quadratic bootstrap mean                                    | -0.120     | -0.12                  | sr09_nonlinearity.json          |
| PASS        | S9.26   | SR 9      | gPS quadratic bootstrap SD                                      | 0.017      | 0.018                  | sr09_nonlinearity.json          |
| PASS        | S9.27   | SR 9      | gPS quadratic bootstrap CI low                                  | -0.152     | -0.154                 | sr09_nonlinearity.json          |
| PASS        | S9.28   | SR 9      | gPS quadratic bootstrap CI high                                 | -0.085     | -0.085                 | sr09_nonlinearity.json          |
| PASS        | S9.29   | SR 9      | lowest bootstrap sign stability (%)                             | 100        | 100.0                  | sr09_nonlinearity.json          |
| PASS        | S9.30   | SR 9      | lowest bootstrap significance rate (%)                          | 100        | 100.0                  | sr09_nonlinearity.json          |
| PASS        | S9.31   | SR 9      | pairs after excluding safety liabilities                        | 8433       | 8433                   | sr09_nonlinearity.json          |
| PASS        | S9.32   | SR 9      | TA quadratic coefficient after that exclusion                   | -0.474     | -0.474                 | sr09_nonlinearity.json          |
| PASS        | S9.33   | SR 9      | fitted peak of the therapeutic-area curve                       | 1.90       | 1.9                    | sr09_nonlinearity.json          |
