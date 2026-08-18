# R2-MJ-14 — directionality checks

Round-1 referee, on the 92.5% fully concordant pleiotropic lead variants: the
figure looks implausibly high; alleles that raise autoimmune risk should lower
infection risk; the alleles at PCSK9 that raise a hypercholesterolaemia
diagnosis should raise type 2 diabetes risk.

We are not defending full concordance — 92.5% is not a claim that antagonistic
pleiotropy is rare. `01_directionality_checks.ipynb` takes two variants where a
trade-off is expected and asks whether the method detects it. No scan, no
modelling.

## Alleles, not variants

Every direction names an **effect allele**: the alternative allele of the
variant identifier `chrom_pos_ref_alt`, which is what studies are harmonised to
at ingestion. Credible sets sharing a `variantId` therefore share an effect
allele and nothing is realigned in the notebook; harmonisation itself is a
Methods statement and is not re-verified. Signed effect is
`rescaledStatistics.directionOfEffect × rescaledStatistics.absEstimatedBeta`,
where `directionOfEffect` is the sign of the study's reported beta
(`src/manuscript_methods/rescaled_beta.py`); `originalBeta` is carried through
every table. Concordance is the Methods definition: largest proportion of
same-direction effects per variant over the credible sets reporting a beta.

## Result in one table

|                                            | PCSK9 `rs11591147`         | TYK2 `rs34536443` | FUT2 `rs601338` (fallback)                    |
| ------------------------------------------ | -------------------------- | ----------------- | --------------------------------------------- |
| variant id                                 | `1_55039974_G_T`           | `19_10352442_G_C` | `19_48703417_G_A`                             |
| **effect allele**                          | **T**                      | **C**             | **A** (non-secretor)                          |
| other allele                               | G                          | G                 | G                                             |
| consequence in 25.06                       | `PCSK9 R46L`               | `TYK2 P1104A`     | `FUT2 W154*` (= W143X, other numbering)       |
| MAF, major LD population                   | 0.039                      | 0.045             | 0.496                                         |
| credible sets / diseases                   | 60 / 21                    | 54 / 19           | 14 / 12                                       |
| credible sets increasing / decreasing risk | 2 / 54                     | 1 / 38            | 11 / 2                                        |
| **concordance (paper formula)**            | **0.964**                  | **0.974**         | **0.846**                                     |
| predicted counter-trait                    | type 2 diabetes            | tuberculosis      | enteric infection                             |
| counter-trait on the variant               | **no**                     | **no**            | **no**                                        |
| counter-trait in its cluster               | no (0 of 62 credible sets) | no (0 of 136)     | yes, but on 7 other lead variants (13 of 102) |

## Check 1 — PCSK9 `rs11591147` (R46L), effect allele T

Hypothesis: the allele that lowers hypercholesterolaemia risk raises type 2
diabetes risk.

The T allele **lowers** risk right across the lipid–coronary axis, which is the
loss-of-function direction and the opposite sign to the premise in the comment
(which assumes an allele that _increases_ a hypercholesterolaemia diagnosis):

- hypercholesterolaemia `HP_0003124`, 7 credible sets: β −0.17 to −0.30
- familial hypercholesterolaemia `EFO_0004911`, 3 credible sets: β −0.38 to
  −0.46
- familial hyperlipidaemia `MONDO_0001336`, 3 credible sets: β −0.37 to −0.51
- coronary artery disease `EFO_0001645`, 10 credible sets: β −0.13 to −0.31

**Type 2 diabetes (`MONDO_0005148`) is not linked**: 4,757 qualifying credible
sets corpus-wide on 3,367 lead variants, **0** on `rs11591147`, **0** anywhere
in its colocalisation cluster (62 credible sets, 2 lead variants, 23 diseases —
lipid, coronary and vascular terms, plus metabolic disease and response to
statin). The prediction is untestable at this locus; that is coverage, not
direction, and no proxy trait was substituted.

Concordance is **0.964**, not 1: two credible sets of `MONDO_0009387` (PheCode
272.12, "Hyperglyceridemia") run positive (+0.27, +0.31) against 54 negative.

## Check 2 — TYK2 `rs34536443` (P1104A), effect allele C

Hypothesis: the allele that protects against autoimmune disease raises
tuberculosis risk.

The C allele is protective in **every** autoimmune association it carries:

| disease                                          | credible sets | β range        |
| ------------------------------------------------ | ------------- | -------------- |
| psoriasis `EFO_0000676`                          | 11 (8 with β) | −0.27 to −0.46 |
| rheumatoid arthritis `EFO_0000685`               | 11 (7 with β) | −0.18 to −0.35 |
| ACPA-positive rheumatoid arthritis `EFO_0009459` | 2             | −0.33, −0.37   |
| systemic lupus erythematosus `MONDO_0007915`     | 2 (1 with β)  | −0.40          |
| type 1 diabetes `MONDO_0005147`                  | 2             | −0.28, −0.31   |
| sarcoidosis `MONDO_0019338`                      | 3             | −0.28 to −0.32 |
| psoriasis vulgaris `EFO_1001494`                 | 4 (1 with β)  | −0.35          |
| hypothyroidism `EFO_0004705`                     | 5             | −0.07 to −0.14 |
| Hashimoto's thyroiditis `EFO_0003779`            | 1             | −0.12          |
| Crohn's disease `EFO_0000384`                    | 1             | −0.26          |
| arthritis `EFO_0005856`                          | 1             | −0.15          |
| autoimmune disease `EFO_0005140`                 | 2             | −0.16, −0.19   |

Ankylosing spondylitis has no credible set on this variant.

**Tuberculosis is not linked.** The whole corpus holds **5** tuberculosis
credible sets, on one term (`MONDO_0018076`) and five other lead variants
(GCST000764, GCST001398, GCST002810, GCST004923, GCST90275067); none is on
`rs34536443` and none is in its cluster (136 credible sets, 19 lead variants).
The hypothesis as stated is untestable, so the fallback variant was run.

**The trade-off is still detectable on this allele, against a different
infection.** The only association of the C allele that runs positive is
**tonsillitis (`MONDO_0001039`), β = +0.19** — the same allele that protects
against autoimmunity raises risk of an infectious condition — and that one
credible set is exactly why the concordance is **0.974** rather than 1. Two
further infection- or metabolism-relevant terms carry no usable direction: three
COVID-19 and two type 2 diabetes credible sets on this variant report no beta.
Tonsillitis is reported on clinical grounds and named explicitly, because
`MONDO_0001039` is not an ontology descendant of `EFO_0005741` (infectious
disease) in the 25.06 release — it sits under disorder of pharynx / upper
respiratory tract disorder.

## Fallback — FUT2 `rs601338`, effect allele A (non-secretor)

Run because tuberculosis is not linked to TYK2. Hypothesis: the non-secretor
allele protects against enteric infection and raises Crohn's disease risk.

**The Crohn's half holds on the named allele**: Crohn's disease `EFO_0000384`, β
= **+0.10**, increases risk. Same direction on type 1 diabetes (+0.11, +0.13),
duodenal ulcer (+0.15), gallstones (+0.08), cholelithiasis (+0.06),
hypercholesterolaemia (+0.05), hyperlipidaemia (+0.05), essential hypertension
(+0.04). Two anaemia terms run the other way — megaloblastic anaemia −0.15,
deficiency anaemia −0.12 — giving concordance **0.846**.

**The enteric-infection half is not testable on this variant**: `rs601338`
carries no infectious-disease association of its own. Its cluster (102 credible
sets, 42 lead variants) does contain 13 infectious-disease credible sets —
intestinal infectious disease, dysentery, viral (intestinal) disease, COVID-19,
peritonsillar abscess, tracheitis — but on **7 other lead variants**, whose
betas are stated with respect to _their_ effect alleles. Those directions cannot
be carried over to the A allele without LD phase, which is not available here,
so they are reported as cluster coverage and not as a direction for `rs601338`.

## What this supports in the response

In all three cases the predicted counter-trait had no credible set on the
variant itself, so the specific predictions could not be evaluated — reported as
coverage rather than proxied. Where an opposing association does exist, the
pipeline registers it with no special handling: both PCSK9 R46L and TYK2 P1104A
come out **below** full concordance, each driven by the single association that
runs against the variant's dominant direction, and FUT2 sits at 0.846. The 92.5%
headline reflects which trait pairs have been measured in the corpus, not a
claim that antagonistic pleiotropy is rare.

## Notebook 02 — immunity versus infection, systematically

`02_immunity_infection_pleiotropy.ipynb` narrows to the axis the referee names
and works in three steps: genes pleiotropic across the two therapeutic areas,
then the lead variants that carry both sides (the only allele-exact comparison),
then a few representatives.

Classes are therapeutic areas from `disease.therapeuticAreas`: `EFO_0000540`
(immune system disease, 107 diseases with data, 9,453 credible-set × disease
rows) and `EFO_0005741` (infectious disease, 57 diseases, 1,143 rows). The
therapeutic-area axis is used rather than the `autoimmune disease` ontology
branch because that branch does not contain type 1 diabetes, celiac disease or
psoriasis.

### Headline

- **207 genes** have at least one credible set in each area; **167** survive
  dropping trans-disease / MTAG studies (see below). The biggest are the
  expected immune regulators — IL12B (107 immune credible sets), STAT4 (102),
  TYK2 (82), IRF5 (81), TNFAIP3 (75) — but nearly all have only one or two
  infection credible sets.
- **Only 59 gene × lead-variant pairs (44 genes, 58 lead variants)** carry both
  areas on the _same_ lead variant. That is the only configuration where the two
  directions refer to the same effect allele: 207 genes of gene-level overlap
  collapse to 44 genes of allele-exact overlap.
- **Trans-disease / MTAG studies matter a lot.** 81 credible sets map to an
  immune _and_ an infection disease at once ("Severe COVID-19 or rheumatoid
  arthritis (MTAG)"), putting the identical beta on both sides. Dropping them
  halves the shared-variant count, 122 → 59.
- **Verdicts on the 59 pairs: 34 concordant, 16 discordant, 9 undetermined — 32%
  of determined pairs discordant**, against 7.5% at the genome-wide variant
  level.

### Representatives (direction always for the named effect allele)

| gene             | variant            | effect allele | immune side                                                                                        | infection side                                                  | verdict                         |
| ---------------- | ------------------ | ------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------- |
| TYK2             | `19_10355447_C_T`  | T             | psoriatic arthritis −0.17                                                                          | COVID-19, 11 credible sets, +0.06 to +0.22                      | discordant                      |
| TYK2 P1104A      | `19_10352442_G_C`  | C             | 38 credible sets, all negative (psoriasis −0.27…−0.46, RA −0.18…−0.35, SLE −0.40, T1D −0.28/−0.31) | 3 COVID-19 credible sets, none with a beta; tonsillitis +0.19   | coverage limit, not concordance |
| C1orf141 / IL23R | `1_67131436_G_A`   | A             | Crohn's disease −0.27                                                                              | leprosy **+0.70**                                               | discordant                      |
| CCDC88B          | `11_64340263_G_A`  | A             | immune system disease −0.13, psoriasis −0.07                                                       | leprosy **+0.38**                                               | discordant                      |
| LACC1            | `13_43883789_A_G`  | G             | Crohn's disease, 5 credible sets, +0.14 to +0.29                                                   | leprosy **+1.11**                                               | concordant                      |
| FLG              | `1_152313385_G_A`  | A             | atopic eczema +0.35/+0.91, contact dermatitis +0.55, allergic disease +0.66                        | dermatophytosis / tinea +0.12 to +0.14                          | concordant                      |
| SH2B3            | `12_111446804_T_C` | C             | 19 credible sets, all negative (T1D, celiac, MS)                                                   | respiratory infection −0.04, prosthesis-related infection −0.03 | concordant                      |
| ABO              | `9_133273813_C_T`  | T             | Graves disease +0.34                                                                               | COVID-19, 8 credible sets, −0.06 to −0.10                       | discordant                      |

The two leprosy loci make the point and its limit at once: IL23R and CCDC88B
show the classic antagonistic pattern, while LACC1 — also a leprosy/Crohn's
locus — is concordant on the same axis.

### Caveats

The infection side is dominated by COVID-19 (49 of the comparable credible sets,
16 lead variants). Several Hodgkin-lymphoma terms carry the infectious-disease
area in 25.06 through their viral aetiology; they are cancers and should not be
read as infection trade-offs. The immune-system area is broader than
autoimmunity by design, so allergy and inflammatory terms are included (FLG's
eczema).

## Notebook 03 — type 2 diabetes versus coronary disease (the PCSK9 axis)

`03_t2d_cad_pleiotropy.ipynb` runs the same three steps on the referee's other
example: genes with credible sets in both classes, the lead variants that carry
both sides, then PCSK9 itself and a few representatives.

Neither class has a usable single ontology root — `MONDO_0005148` (type 2
diabetes) has 2 descendants, and MI, angina and CABG are not descendants of
`EFO_0001645` — so both are explicit root lists plus descendants, in a primary
and a broad version. Broad additions are taken as **exact terms** where a
closure would overreach (the closure of `EFO_0000400` contains type 1 and
monogenic diabetes; the closure of `EFO_0000319` contains atrial fibrillation,
hypertension and stroke). Type 1 diabetes, gestational and CF-related diabetes
are excluded throughout.

### PCSK9 — the question cannot be asked here

109 qualifying credible sets across **24 lead variants** have PCSK9 as their
L2G-prioritised gene. **Zero** are mapped to type 2 diabetes or to any glycaemic
term, under either class definition. Its coronary side is well populated (40
credible sets primary, 48 broad) and points the protective way for the
LDL-lowering alleles, alongside the lipid terms (hypercholesterolaemia 15,
hyperlipidaemia 8, familial hyperlipidaemia 7).

**What the Platform shows, and why our corpus does not.** The full 25.06 release
does carry one diabetes-mapped credible set on `rs11591147` itself:
**GCST90309362**, "Diabetes (confirmatory factor analysis Factor 28)" (Carey et
al. 2024), `EFO_0000400` diabetes mellitus, β = **−0.0092** for the **T**
allele, p = 8.1 × 10⁻¹¹, SuSiE-inf, L2G-assigned to PCSK9. It is excluded at
**study** level, not variant level: the study reports **0 cases and 0 controls**
on 360,514 samples, so it fails the `binaryLessCases` requirement — it is a
continuous latent factor score, not a case/control diabetes diagnosis, and all
34 Carey factor studies in the release are dropped the same way. Even so, it
does not support the predicted trade-off: the T allele, which lowers
hypercholesterolaemia and coronary risk, also has a **negative** effect on the
diabetes factor. Searching the whole release across PCSK9 ± 500 kb finds only
one other diabetes-mapped credible set, `1_55407443_T_C` (GCST90296598, 291
cases), 370 kb away and L2G-assigned to USP24, also outside the qualifying
corpus.

So the predicted trade-off is untestable at PCSK9 here: no case/control diabetes
credible set exists at the locus — missing data, not a concordant result.

### Genes and shared lead variants

- **160 genes** (205 broad) have credible sets in both classes — TCF7L2, CDKN2B,
  CDKAL1, IGF2BP2, FTO, HHEX, CCND2 lead.
- Only **27 gene × lead-variant pairs (23 genes)** carry both classes on the
  same lead variant; 37 pairs (29 genes) under the broad classes.
- 90 credible sets come from four trans-disease studies ("Type 2 diabetes
  mellitus or coronary artery disease (pleiotropy)", "Cardiometabolic
  multimorbidity", "Coronary heart disease × type 2 diabetes interaction") and
  are excluded from the direction comparison.
- **Verdicts:** primary 15 concordant / 3 discordant / 9 undetermined (16.7% of
  determined discordant); broad 25 / 4 / 8 (13.8%). Unlike the
  immunity–infection axis, concordance dominates here — cardiometabolic risk
  mostly moves together.

### Representatives (direction for the named effect allele)

| gene                     | variant            | effect allele | diabetes side                                                                                  | coronary side                                                              | verdict        |
| ------------------------ | ------------------ | ------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------- |
| APOE                     | `19_44908684_T_C`  | C             | 16 credible sets, median −0.06 (T2D, diabetes mellitus, diabetic retinopathy, T2D nephropathy) | 14 credible sets, median +0.08 (MI, CAD, angina, coronary atherosclerosis) | **discordant** |
| APOE                     | `19_44919689_A_G`  | G             | 1 credible set, −0.06                                                                          | 5 credible sets, +0.06                                                     | **discordant** |
| PNPLA3                   | `22_43928850_C_T`  | T             | 3 credible sets, +0.06                                                                         | 2 credible sets, −0.03                                                     | **discordant** |
| CUX2 (chr12 SH2B3/ALDH2) | `12_111269073_C_T` | T             | 1 credible set, +0.03                                                                          | 4 credible sets, −0.13                                                     | **discordant** |
| TCF7L2                   | `10_112998590_C_T` | T             | 76 credible sets, +0.25                                                                        | 8 credible sets, +0.07                                                     | concordant     |
| CDKN2B (9p21)            | `9_22125504_G_C`   | C             | 1 credible set, +0.07                                                                          | 16 credible sets, +0.16                                                    | concordant     |
| HNF1A                    | `12_120978847_A_C` | C             | 2 credible sets, +0.04                                                                         | 11 credible sets, +0.05                                                    | concordant     |
| LPL                      | `8_19973410_C_T`   | T             | 2 credible sets, −0.04                                                                         | 1 credible set, −0.07                                                      | concordant     |

**APOE is the mechanism the referee describes, found without special handling**:
the allele that raises coronary risk lowers type 2 diabetes risk, and the
variant's paper-formula concordance is **0.659** — among the lowest of any
highly pleiotropic lead variant (191 credible sets).

### Caveats

The primary coronary class inherits `EFO_0010820` (spontaneous coronary artery
dissection, 29 credible sets) as a descendant of coronary artery disease —
different pathology. The diabetes side is overwhelmingly one term (type 2
diabetes, 4,757 credible sets), so this is a diagnosis axis; the glycaemic and
lipid _measurements_ that would show the LDL–insulin-secretion trade-off most
directly sit outside this disease-only corpus.

## Outputs

All in `data/intermediate_files/`, suffixed `-r1`; no published figure or
non-`-r1` file touched.

| file                                                  | contents                                                                                                                                          |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `directionality_variant_checks-r1.csv`                | one row per variant: effect allele, MAF, credible-set and disease counts, direction counts, concordance, hypothesis, verdict                      |
| `directionality_variant_associations-r1.csv`          | every disease association of the three variants, with effect allele, other allele, `originalBeta`, rescaled β and direction                       |
| `directionality_cluster_coverage-r1.csv`              | per variant: cluster size, and whether the predicted counter-trait appears in the cluster and on which variant                                    |
| `immunity_infection_genes-r1.csv`                     | the 207 genes with credible sets in both areas, with credible-set, disease and lead-variant counts                                                |
| `immunity_infection_shared_variants-r1.csv`           | the 59 gene × lead-variant pairs carrying both areas on one allele, with signs, median betas, verdict and the variant's paper-formula concordance |
| `immunity_infection_representatives-r1.csv`           | every association of the 8 representatives, association by association                                                                            |
| `t2d_cad_genes-r1.csv`                                | the 160 genes with credible sets in both the diabetes and coronary classes                                                                        |
| `t2d_cad_shared_variants-r1.csv` / `..._broad-r1.csv` | gene × lead-variant pairs carrying both classes on one allele, primary and broad class definitions, with signs, median betas and verdict          |
| `t2d_cad_pcsk9_profile-r1.csv`                        | every disease PCSK9 is linked to as L2G gene, with class labels and direction counts                                                              |
| `t2d_cad_pcsk9_release_diabetes-r1.csv`               | the diabetes-mapped credible sets at PCSK9 ± 500 kb in the **whole** 25.06 release, with case/control counts and corpus membership                |
| `t2d_cad_representatives-r1.csv`                      | every association of the 8 representatives on this axis                                                                                           |

## Running it

```sh
cd chapters/06-review-r1/directionality-checks
../../../.venv/bin/python -m nbconvert --to notebook --execute --inplace 01_directionality_checks.ipynb
../../../.venv/bin/python -m nbconvert --to notebook --execute --inplace 02_immunity_infection_pleiotropy.ipynb
../../../.venv/bin/python -m nbconvert --to notebook --execute --inplace 03_t2d_cad_pleiotropy.ipynb
```

Notebooks 02 and 03 additionally read
`data/intermediate_files/list_of_prioritised_genes_per_CS.parquet` and
`data/25.06/output/target`, and need no colocalisation scan.

pandas/pyarrow only, no Spark; about a minute. Inputs:
`data/intermediate_files/qualifying_credible_sets`,
`data/25.06/output/{disease,variant,colocalisation_coloc,colocalisation_ecaviar}`
(colocalisation scanned for chromosomes 1 and 19 only, to build the three
clusters).
