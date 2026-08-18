# Supplementary Figure SR6 — unique diseases vs unique TAs per colocalisation cluster

Reviewer 1, minor comment 9. The manuscript reports Spearman ρ = 0.81, P <
1×10⁻¹⁶ between the number of unique diseases per colocalisation cluster and the
number of unique therapeutic areas (TAs) per cluster
(`sections/results/04_variant_pleiotropy.tex`, para 1). The reviewer asked for
the underlying data as a supplementary figure.

## Files

| File                                                                           | What                                                                                                     |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `cluster_lib_r1.py`                                                            | Shared clustering primitives. Asserts its TA hierarchy against notebook cell 18.                         |
| `00_build_cluster_table-r1.py`                                                 | Builds the cluster-level table. Run from the repo root.                                                  |
| `01_cluster_disease_vs_ta-r1.ipynb`                                            | Control asserts + column comparison + SR6 figure. Executed, no error cells.                              |
| `figure_sr6-r1.pdf`                                                            | Supplementary figure SR6.                                                                                |
| `02_st15_cluster_membership-r1.py`                                             | Builds Supplementary Table 15 (never previously built).                                                  |
| `03_temporal_vps-r1.py`                                                        | Temporal vPS for Figure 4a.                                                                              |
| `04_figure_3_resolved-r1.py`                                                   | Figure 3a/b data.                                                                                        |
| `../../../data/intermediate_files/cluster_disease_ta_counts-r1.csv`            | Source table, 20,041 rows.                                                                               |
| `../../../data/intermediate_files/cluster_disease_ta_column_comparison-r1.csv` | raw vs resolved side by side.                                                                            |
| `../../../data/intermediate_files/cluster_disease_ta_figure_numbers-r1.csv`    | Numbers plotted in SR6.                                                                                  |
| `../../../data/intermediate_files/st15_cluster_membership-r1.csv`              | Supplementary Table 15, 42,918 rows.                                                                     |
| `../../../data/intermediate_files/cluster_stats_by_year-r1.csv`                | Temporal vPS, both columns.                                                                              |
| `../../../data/intermediate_files/cluster_stats_by_year_column_diff-r1.csv`    | Year-by-year delta.                                                                                      |
| `../../../data/intermediate_files/figure_3_column_diff-r1.csv`                 | Figure 3 coefficient delta.                                                                              |
| `../../figure_3/data/plot_a-r1.csv`, `plot_b-r1.csv`                           | Figure 3a/b data, resolved column.                                                                       |
| `../../figure_3/figure_3-r1.R`                                                 | `figure_3.R` with the two input paths and the output path swapped to `-r1`. Six lines differ, all paths. |
| `../../figure_3/figure_3_final-r1.pdf`                                         | Figure 3, rebuilt.                                                                                       |

Nothing published was overwritten. Run the Python scripts from the repository
root; run `figure_3-r1.R` from `chapters/03-manuscript-figures` so `renv`
activates.

## Source table and its provenance

The two counts are `uniqueTraitsInCluster` (unique diseases) and
`clusterNumberTherapeuticAreas` (unique TAs), both defined in
`chapters/02-analysis/04-variant-level-ps/02_clustering_analysis.ipynb`.

**No cluster-level table existed on disk.** That notebook holds
`cluster_pleiotropy` only in a Spark session and never writes it; a search of
every `.parquet`/`.csv` in the repo found no table carrying either column. So
`00_build_cluster_table-r1.py` re-derives it from the unchanged release inputs
using that notebook's exact logic, reimplemented on pyarrow instead of Spark:

- cell 5 — `qualifying_credible_sets` sorted by p-value then lead-variant PIP;
  colocalisation edges from `colocalisation_coloc` (h4 ≥ 0.8) and
  `colocalisation_ecaviar` (clpp ≥ 0.01), both endpoints restricted to
  qualifying study loci;
- cell 6 — `cluster_lead_variants`, verbatim: connected components over
  colocalisation edges plus shared-lead-variant edges;
- cell 13 — unique-disease count = distinct trait ids flattened over cluster
  members;
- cells 18–19 — disease → primary TA via `disease.parquet` `ancestors` matched
  against the 23-entry `therapy_area_hierarchy` in order, unmatched → `"other"`,
  lookup semi-joined against `study.diseaseIds`; TA count = size of the distinct
  mapped set.

The cluster partition is a connected-components result, so it is independent of
traversal order and identical to the notebook's. Cell 19's inner join onto
`variants_pleiotropy` is omitted — that intermediate is no longer on disk, and
the notebook's own output shows the join was not restrictive
(`cluster_pleiotropy.count()` = `result_df.count()` = 20,041), so it cannot
change either count.

Inputs used: `data/intermediate_files/qualifying_credible_sets`,
`data/25.06/output/{colocalisation_coloc, colocalisation_ecaviar,disease,study}`.

## Control result — PASS

Run on the `_raw` counts, i.e. the `traitFromSourceMappedIds` column the
clustering notebook actually reads. Every number the manuscript states
reproduces, and all six statistics for which the notebook printed full precision
match to the last digit:

| Quantity                 | Reproduced                          | Published / notebook |
| ------------------------ | ----------------------------------- | -------------------- |
| Clusters                 | 20,041                              | 20,041               |
| CSs clustered            | 70,618                              | 70,618               |
| Clusters of size >1      | 8,136                               | 8,136                |
| Clusters with >1 disease | 6,678                               | 6,678                |
| Fraction >1 disease      | 0.3332169053440447                  | 0.3332169053440447   |
| Unique diseases, range   | 1–122                               | 1–122                |
| Unique diseases, mean    | 2.156080035926351                   | 2.156080035926351    |
| Clusters with >1 TA      | 4,536                               | 4,536                |
| Fraction >1 TA           | 0.22633601117708696                 | 0.22633601117708696  |
| Unique TAs, max          | 20                                  | 20                   |
| Unique TAs, mean         | 1.4024749264008782                  | 1.4024749264008782   |
| **Spearman ρ**           | **0.8071208888491556**, P < 1×10⁻¹⁶ | 0.8071208888491556   |

This confirms the table has the same lineage as the published figures. The
figure itself is then drawn from the _resolved_ column, for the reason below.

## Trait-column inconsistency in the clustering code path

`qualifying_credible_sets` carries two trait columns:

- `traitFromSourceMappedIds` — the raw curator-supplied mapping;
- `diseaseIds` — the ontology-resolved mapping. Equal to the study index's
  `diseaseIds` for **all 70,618 CSs**.

They agree on 69,962 CSs (99.07%) and differ on 656. `diseaseIds` is never
larger than the raw set and is smaller in 378 CSs, because two raw ids can
collapse onto one MONDO term.

**Every other analysis in the repo uses `diseaseIds`** — variant-level
`uniqueDiseases`
(`02-analysis/04-variant-level-ps/01_variant_level_pleiotropy.ipynb` cell 7),
gene-level gPS (`02-analysis/05-gene-level-ps/01_gene_level_pleiotropy.ipynb`),
`mappedTherapeuticAreas`
(`01-data-preparation/04_qualifying_dataset_generation.ipynb` cell 8), and all
the target-enrichment work. `traitFromSourceMappedIds` appears only in the
clustering code path, copy-pasted across four files:

- `02-analysis/04-variant-level-ps/02_clustering_analysis.ipynb` cell 5
- `02-analysis/05-gene-level-ps/02_temporal_vPS_gPS.ipynb` cell 14
- `03-manuscript-figures/figure_3/python_scripts/clustering_analysis.ipynb` cell
  5
- `03-manuscript-figures/figure_3/python_scripts/prepare_plot_a_b_data.py`

So vPS (cluster-level, raw column) and lead_vPS (variant-level, resolved column)
were counted on different columns. That is not deliberate.

### What the raw column costs

26 of the 1,425 distinct raw trait ids in the qualifying CSs have **no row in
`disease.parquet` at all**:

- retired EFO ids — `EFO_0000493` (302 CSs), `EFO_0000516` glaucoma (143),
  `EFO_0003761` (97), `EFO_0000352` (48), 9 more;
- Orphanet ids Open Targets never ingested — `Orphanet_797` sarcoidosis (18),
  `Orphanet_34533` (6), 5 more;
- HP terms — `HP_0000735`, `HP_0100601`;
- junk — `HANCESTRO_0014` (an _ancestry_ term, from "Asthma x Hispanic
  interaction (2df)") and the literal string `gastric adenocarcinoma` in an id
  field.

Cell 18's UDF drops any id missing from the lookup:

```python
mapped_areas.append(lookup_dict.get(efo_id, None))
mapped_areas = list(set(area for area in mapped_areas if area is not None))   # silently dropped
```

so those clusters get an **empty** TA array and `f.size` = 0. Note this is _not_
the `"other"` bucket — `"other"` is applied when building `efo_ta` and only
covers terms that _are_ rows in `disease.parquet` but match none of the 23
hierarchy ancestors. It cannot rescue an id that never got a row. Meanwhile the
disease count reads the raw ids with no lookup at all, which is how a cluster
ends up with 1 disease and 0 TAs. Open Targets had already remapped all 26 to
live MONDO terms in `diseaseIds` (`EFO_0000516` → `MONDO_0005041`, `EFO_0003761`
→ `MONDO_0002009`, `Orphanet_797` → `MONDO_0019338`, …).

33 clusters (0.16%) end up with 0 TAs on the raw column, attributable as:
`EFO_0000516` 21, `Orphanet_34533` 4, `EFO_0003761` 4, and one each from
`EFO_0004215`, `Orphanet_797`, `Orphanet_478`, `EFO_0003847`. 31 are singletons,
2 have size 2. Separately, 656 of 70,618 CSs touch an unresolvable id, so some
clusters that do get a TA count are silently one TA short — one-directional, and
unrecoverable from the raw column since the dead terms have no ancestors to
consult.

### Effect of switching to `diseaseIds`

|                          | raw (published) | resolved (plotted)  |
| ------------------------ | --------------- | ------------------- |
| Clusters                 | 20,041          | 20,041              |
| Clusters with >1 disease | **6,678** (33%) | **6,617** (33%)     |
| Unique diseases, range   | **1–122**       | **1–120**           |
| Unique diseases, mean    | **2.1561**      | **2.1415**          |
| Clusters with >1 TA      | 4,536 (23%)     | 4,539 (23%)         |
| Unique TAs, range        | **0–20**        | **1–20**            |
| Unique TAs, mean         | 1.4025          | 1.4046              |
| Clusters with 0 TAs      | 33              | **0**               |
| Spearman ρ               | 0.807121 → 0.81 | 0.813043 → **0.81** |
| At (1 disease, 1 TA)     | 13,330          | 13,424              |

The zero-TA clusters disappear, so the TA range becomes 1–20 exactly as the
manuscript already states — the text was right and the code was wrong. ρ still
rounds to 0.81, both percentages still round to 33% and 23%, and no conclusion
changes.

### Manuscript edits this requires

`sections/results/04_variant_pleiotropy.tex`, lines 10–12, currently:

> Across these clusters, 6,678 (33\%) were linked to multiple diseases (range
> 1--122, mean 2.16), and 4,536 (23\%) were linked to multiple TAs (range 1--20,
> mean 1.40).

should become **6,617**, **range 1--120**, **mean 2.14**, **4,539**. The
percentages, the TA range and the TA mean are unchanged. Not applied here —
`manuscript_gentropy` is read-only from this repo.

Figure 3 and the temporal vPS/gPS analysis are drawn from the same clustering
code path and so carry the same raw-column counts; they were not rebuilt as part
of this task.

## Overplotting treatment

Both counts are small integers over 20,041 clusters, so a naive scatter is a
solid block at the origin with a sparse tail to 120. Treatment:

- **Nothing is subsampled and no jitter is applied.** The 20,041 clusters
  collapse onto 219 distinct integer coordinates; each coordinate is drawn as
  one marker.
- **Marker area ∝ √n**, where _n_ is the number of clusters at that coordinate
  (`s = 6·√n` pt², i.e. marker radius ∝ n^¼). Area ∝ _n_ was rejected: the modal
  coordinate holds 13,424 clusters and would have swamped the panel. A size key
  on the right gives the mapping at n = 1, 10, 100, 1,000 and 13,424.
- `alpha=0.75` with a 0.3 pt white marker edge, so adjacent and overlapping
  markers stay separable.
- It remains a scatter — no heatmap, contour or 2-D binning.

## The figure

One row, two panels, same data and same points, differing only in axis scaling.

- **a** — linear axes, unique diseases (x) against unique TAs (y). Carries the ρ
  annotation.
- **b** — the same, both axes log-scaled.

Both counts have a minimum of 1 on the resolved column, so both panels plot all
20,041 clusters and there are no zeros to handle on the log axes. ρ is annotated
on panel a only, so the caption should not repeat it.

Styling matches `figure_sr5.pdf` (built by
`../effective-independent-traits/03_ninepanel_nonlinearity.ipynb`): matplotlib
default DejaVu Sans, axis labels 9 pt, panel titles 10 pt with `pad=8`, legend 8
pt frameless, dotted grid at `alpha=0.3`, top and right spines hidden,
`savefig(bbox_inches="tight", dpi=150)`. Panel letters use the repo convention
from `ed8_translation_success_by_pleiotropy.ipynb`:
`ax.text(-0.06, 1.04, tag, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")`.

Page width: SR5 is 14.89 in wide across 3 panel columns and is included at
`0.74\textwidth` of a 6.75 in text block, giving 1.67 in per panel. SR6 is 10 in
wide across 2 panel columns, so
`\suppfig{figures/figure_sr6.pdf}{0.5\textwidth}` reproduces the same on-page
panel size and therefore the same rendered font sizes.

## Exact numbers plotted

| Quantity                          | Value                                                                           |
| --------------------------------- | ------------------------------------------------------------------------------- |
| Clusters plotted, both panels     | 20,041                                                                          |
| Distinct coordinates              | 219                                                                             |
| **Clusters at (1 disease, 1 TA)** | **13,424 (66.98%)**                                                             |
| Clusters with 0 TAs               | 0                                                                               |
| Unique diseases: min / max / mean | 1 / 120 / 2.1415                                                                |
| Unique TAs: min / max / mean      | 1 / 20 / 1.4046                                                                 |
| Clusters with >1 disease          | 6,617                                                                           |
| Clusters with >1 TA               | 4,539                                                                           |
| Spearman ρ                        | 0.813043                                                                        |
| Spearman P                        | underflows to 0.0 in double precision; the manuscript's P < 1×10⁻¹⁶ bound holds |

Ten most populated coordinates (disease count, TA count → clusters):

(1,1) → 13,424 · (2,2) → 1,547 · (2,1) → 1,524 · (3,2) → 688 · (3,3) → 311 ·
(3,1) → 297 · (4,2) → 297 · (4,3) → 171 · (5,2) → 132 · (5,3) → 123

Together 18,514 clusters, 92.4% of the total, which is why the size encoding is
necessary.

---

# Downstream consumers of the raw column

Seven code sites read `traitFromSourceMappedIds`. Three needed rebuilding, four
were checked and cleared. Every rebuild reproduces its published artefact on the
raw column first; only then is the resolved column reported.

## Rebuilt

### Supplementary Table 15 — `02_st15_cluster_membership-r1.py`

Never previously built (`extended_data.tex:136` carried
`% TODO(data): populate tab:st15`). One row per (cluster, disease): 42,918 rows
over 20,041 clusters, **1,403 distinct diseases** — matching the manuscript's
own stated disease count exactly — and 23 therapeutic areas, vPS 1–120.

Controls, both PASS for all 20,041 clusters: rows per cluster equals
`uniqueTraitsInCluster_resolved`, and distinct TAs per cluster equals
`clusterNumberTherapeuticAreas_resolved`. So the table cannot disagree with SR6.
Zero null disease labels and zero null therapeutic areas — no unresolvable id
can leak in.

### Figure 4a, vPS line — `03_temporal_vps-r1.py`

Re-clusters per year for 2006–2025. **Control PASS on all four fields**
(`n_count`, `mean`, `sd`, `se`) against the committed
`cluster_stats_by_year.csv` across all 20 years.

Resolved shift: identical for 2006–2015, worst case −0.688% (2024). Mean vPS in
2025 goes 2.156080 → 2.141510. The line is unchanged for practical purposes. The
gPS line was already on `diseaseIds` and is untouched.

### Figure 3a/b — `04_figure_3_resolved-r1.py`

**Control PASS on 11 fields to ~1e-12** against the committed `plot_a.csv` and
`plot_b.csv` — all 5 statistics × 12 coefficient rows, and all 6 statistics × 7
MAF bins — plus all four R² values the manuscript quotes:

| R²                              | reproduced (raw) | resolved | published |
| ------------------------------- | ---------------- | -------- | --------- |
| Full joint model                | 17.7322%         | 17.6664% | 17.7%     |
| Without predicted power         | 6.0022%          | 5.9622%  | 6.0%      |
| Predicted power alone           | 14.7145%         | 14.6690% | 14.7%     |
| Max effective sample size alone | 0.4450%          | 0.4411%  | 0.45%     |

All four still round to the published values on the resolved column. Univariate
coefficients shift by at most 0.0085 absolute (0.45% relative); MAF-bin observed
means by at most 0.027. `figure_3.R` detects the schema swap in its two inputs
and corrects for it internally, so the `-r1` script needed no change beyond the
paths.

Two vintages of this regression exist in the repo and **the committed figure
data comes from the older one**:
`figure_3/python_scripts/clustering_analysis.ipynb` printed R² = 0.17732233,
while `02-analysis/04-variant-level-ps/02_clustering_analysis.ipynb` printed
0.17975229. The manuscript's 17.7% and 6.0% match the figure vintage, which is
what this script reproduces. The 02-analysis notebook's printed coefficients
(`maxAbsBeta` 2.151716, `predictedPower` 1.444744) are stale relative to the
published figure. Also note `prepare_plot_a_b_data.py:188` points at a GCS copy
of `variant_pleiotropy`; the local `figure_3/python_scripts/variant_pleiotropy`
turns out to be that same vintage, which is why the control matches to 1e-12.

## Checked and cleared

- **`02-variant-effects/01_concordanc_of_lead_variant_diseases.ipynb`** — groups
  on the raw array as a key, and is the source of
  `supplementary_results.tex:79-81` ("16% ... of which only 15% showed
  significant Cochran's heterogeneity"). Recomputed over the 63,593 rows:
  **15.91% raw vs 16.05% resolved**; multi-study pairs 8,051 → 8,096 (+0.6%).
  Both round to 16%. No action.
- **`08-genecorrs/ldsc_rg_representative_manifest.py`** — keys representative
  selection on the raw column, but cleared during the genetic-correlation
  rebuild: the rebuilt 1,114-disease matrix is keyed on `diseaseIds` from the
  study index, not on the manifest's cell keys.
- **`01-data-preparation/02_lead_variant_effect_dataset_preparation.ipynb` cells
  8, 24** — carries both columns through into `lead_variant_effect` and
  `qualifying_credible_sets`. No keying. This is why both columns are available
  downstream; not a bug.
- **Supplementary Figure SR2** (clusters by MAF bin) — cluster count is
  column-independent; only the representative variant can change its bin. **25
  of 20,041 representatives change (0.125%)**, and no bin moves by more than 2
  clusters (largest: 0.4–0.5, 4,067 → 4,065). Far inside the 95% CIs the caption
  describes. No action. SR2 is a PNG with no generating script found, so this is
  the quantity the caption describes computed both ways, not an assert against
  the published bin heights.

## Already on `diseaseIds`, verified unchanged

- **Supplementary Results, "Variant-level pleiotropy modelling"** — the text
  states "40,706 disease-associated variants and 1,403 diseases", and 1,403 is
  exactly the distinct-id count of `diseaseIds` (raw gives 1,425). Reproduced:
  40,706, 1,403, 9,828 (24%), mean 1.4796, max 85.
- **R2-MJ-14 directionality work** — PCSK9 rs11591147 cluster 62 credible sets /
  23 diseases; APOE `19_44908684_T_C` 191 credible sets / 85 diseases / 15 TAs /
  concordance 0.659; immune-infection 59 gene × lead-variant pairs with verdicts
  34/16/9. All verified. For both PCSK9 (23) and APOE (85) the published figure
  equals the **resolved** count, not the raw one.
- gPS and all gene-level work, Supplementary Table 2, Figure 3c, all
  target-enrichment and drug analyses, the extended data figures.

## Numbers that move, all of them

`sections/results/04_variant_pleiotropy.tex`:

| Line | Current      | Corrected        |
| ---- | ------------ | ---------------- |
| 10   | 6,678        | **6,617**        |
| 11   | range 1--122 | **range 1--120** |
| 11   | mean 2.16    | **mean 2.14**    |
| 11   | 4,536        | **4,539**        |
| 35   | 5,183        | **5,188**        |

Unchanged: 33\%, 23\%, TA range 1--20, TA mean 1.40, 20,041 clusters, 5,595
(28\%), 135 lead variants with lead_vPS $\geq$ 10, 31 discordant, 34 genes, and
all four R² percentages.

## Two wording bugs found along the way, independent of the column question

1. **`04_variant_pleiotropy.tex:35`** — "4,794 (92.5\%) showed fully concordant
   directionality ... the remaining 389 (7.5\%) displayed at least one opposing
   direction". The code behind 389 is
   `filter(uniqueDiseases > 1).filter(betaSignConcordance < 0.8)`, so the
   threshold is 0.8, not full concordance; and 4,794 is just `5,183 − 389`,
   which silently absorbs **619 lead variants that have no concordance value at
   all** (single-study variants, null `averageBetaSign`). The true split of the
   5,183 is: 389 below 0.8 (7.5%), 4,175 at or above 0.8 (80.6%), 619 undefined
   (11.9%). A variant at concordance 0.9 does have an opposing direction yet is
   counted among the "fully concordant".
2. **`04_variant_pleiotropy.tex:11`** — the TA range is stated as 1--20, which
   is correct for `diseaseIds` and wrong for the raw column the code used
   (minimum 0). Fixed by the column change itself; no edit needed once the
   counts are rebuilt.

## Two published numbers that could not be reproduced at all

Both pre-existing and unrelated to the column work:

- **`supplementary_results.tex:243`, "1,793 (18\%) showed concordance $<$ 1"**
  among the 9,828 pleiotropic variants. The variant table gives 1,253 (12.7%),
  and no threshold reproduces 1,793: `<1` 1,253, `≤0.8` 1,024, `<0.8` 960,
  `<0.9` 1,167, `≤0.9` 1,178, and the absolute ceiling including all 828 nulls
  is 2,081. The same column reproduces 389 / 31 / 34 exactly, so the column is
  the published one.
- **`supplementary_results.tex:260-267`, the Gaussian-mixture block** — 118
  variants, 100 (85%) two-component, 14.5× ratio, median 7.7, ~22% large-effect.
  **There is no mixture-model code anywhere in the repository**: no
  `GaussianMixture`, no AIC/BIC, no `n_components` in any notebook or script.
  The table gives 197 variants with $\geq$ 10 diseases, not 118. Unverifiable
  from what is on disk.

---

# Figure 4a — regenerated; full Figure 4 is blocked

`05_figure_4a_inputs-r1.py` (repo root) and
`../../03-manuscript-figures/figure_4_panel_a-r1.R` (run from
`chapters/03-manuscript-figures`).

## Blocked: the three-panel figure cannot be rebuilt

`figure_4.R:213` hard-reads `data/figure_4/gene_pleiotropy_by_category.csv` for
panel c. That file **does not exist anywhere on disk and has no producer in the
repository** — the only files mentioning it are its two consumers, `figure_4.R`
and `plot_d.R`. `data/figure_4/` itself did not exist and had to be staged from
`data/intermediate_files/`; four of the five inputs were there,
`gene_pleiotropy_by_category.csv` was not.

So **no `figure_4_final-r1.pdf` was written**. Naming a panel-a-only render that
way would imply the whole figure had been reproduced. Panel a is the only panel
the trait column can touch — panel b is a gene-level NB regression and panel c
is the gene-set enrichment — so panel a alone is rendered, twice, for
comparison:

| File                                   | vPS input                                                |
| -------------------------------------- | -------------------------------------------------------- |
| `figure_4/figure_4_panel_a_raw-r1.pdf` | committed `Fig4A_stats_variant_pleiotropy.csv` (control) |
| `figure_4/figure_4_panel_a-r1.pdf`     | `Fig4A_stats_variant_pleiotropy-r1.csv` (resolved)       |

To finish the full figure, `gene_pleiotropy_by_category.csv` has to be recovered
or regenerated first — a gene-set enrichment table (label, log odds ratio, log
CI bounds, p value) that would need its own control.

## Control — PASS

`Fig4A_stats_variant_pleiotropy.csv` is `cluster_stats_by_year.csv` written
tab-separated with an index column. The regenerated raw column was asserted
against **the figure's own input file**, not just against
`cluster_stats_by_year.csv`, on `n_agg`, `n_count`, `mean`, `sd` and `se` — PASS
on all five for all shared years.

The control also caught a shape difference: the committed figure input covers
**2006–2024 (19 rows)** while `cluster_stats_by_year.csv` runs to 2025 (20
rows), matching the panel's `coord_cartesian(xlim = c(2006, 2024))`. The
resolved file is written over the same 19 years so it is a drop-in replacement.
`Fig4A_stats_gene_pleiotropy.csv` and `Fig4A_stats_gene_coverage.csv` are copied
unchanged — gPS and the coverage line already read `diseaseIds`.

Rendering the control panel and cropping panel a out of the published
`figure_4_final.pdf` gives the same curves, axes, ticks, legend placement and
tag, confirming the panel-a port is faithful.

## Visual diff, raw vs resolved

| Measure                                                | Value                                      |
| ------------------------------------------------------ | ------------------------------------------ |
| Differing pixels (1400 × 1729 render, threshold 8/765) | 5,479 of 2,420,600 — **0.226%**            |
| Bounding box of all difference                         | x 809–1317, y 573–705                      |
| vPS line vertical shift                                | mean **+0.13 px**, max +5 px in one column |
| … over the last 15% of the x-range (~2022–2024)        | mean **+0.74 px**                          |
| vPS line stroke width                                  | 3 px                                       |

Every differing pixel lies on the vPS line in the upper sub-panel from ~2019
rightward. The gPS line, the variants-per-gene sub-panel, both axes, the tick
marks and both legends are pixel-identical. The largest movement, 0.74 px in the
2022–2024 region, is **a quarter of the line's own stroke width** — the line
moves less than its thickness, so the change is not perceptible at any print
size. Mean vPS in 2024 goes 2.151017 → 2.136220 (−0.688%), the worst of any
year; 2006–2015 are identical to the digit.

`data/intermediate_files/figure_4a_column_diff-r1.csv` holds the per-year
raw/resolved/delta table.

---

# Panel c's input IS recoverable — `06_panel_c_recovery-r1.py`

`04_gene_pleiotropy_by categories.ipynb` (note the space in the filename) builds
exactly the table `figure_4.R:213` expects, as `results_df`, with columns
`label`, `odds_ratio`, `log_odds_ratio`, `ci_lower`, `ci_upper`, `log_ci_lower`,
`log_ci_upper`, `p_value`, sorted ascending by `log_odds_ratio` — matching
`figure_4.R:215`. **It never writes it**: there is no `to_csv` in the notebook,
only an inline matplotlib render. That is why a content grep for
`gene_pleiotropy_by_category` found only the two consumers. The filename space
was not the reason.

`03_average_gPS_gene_categories.ipynb` is a different analysis — mean gPS per
category with permutation p-values (`p_value_perm`) — and does not produce the
log-odds table.

## What had to be ported

The notebook needs `{output_path}genes_pleiotropy`, also absent from disk,
written by `01_gene_level_pleiotropy.ipynb` cell 13. Two of that notebook's
inputs are unavailable here — `{release_path}target_prioritisation` (missing)
and `/users/dc16/data/releases/25.06/...` (a hardcoded path from another
machine) — but both feed columns appended in cells 10-11, _after_ the cell-9
aggregation. Panel c consumes only `geneId` and `uniqueDiseases`, so neither
blocks it.

`06_panel_c_recovery-r1.py` therefore ports, Spark-free: notebook 01 cells 3/6/9
restricted to those two columns, then notebook 04 cells 4-11 in full.

## Control — PASS, 42/42 values and 21/21 row order

Panel c of the published `figure_4_final.pdf` is vector text, so
`pdftotext -layout` recovers its 21 row labels with the "Genes" and "In set"
columns that `figure_4.R:250-252` parses out of notebook 04's
`f"{category} ({total}/{pct:.1f}%)"`. Asserted:

- **21/21 category gene counts** exact (368, 557, 861, 440, 1022, 4182, 4520,
  3814, 1137, 214, 785, 419, 166, 1862, 321, 4445, 415, 766, 1489, 4526, 830)
- **21/21 overlap percentages** exact to 0.1 pp
- **row ordering identical** to the published forest plot, including the
  0.0006-wide inversion between Cell non-essential (−0.14987) and Cell essential
  (−0.15054)
- **8,285 genes** with a gPS, matching notebook 01 cell 12's printed output
- **4,445 human-knockout genes**, matching notebook 04 cell 6's printed output

Column contract: the recovered table supplies every input `figure_4.R` reads —
`label`, `log_odds_ratio`, `log_ci_lower`, `log_ci_upper`, `p_value`. `fdr` is
computed by the script itself at line 225 (`p.adjust`), and `label_display`,
`n_label`, `pct_label`, `sig_label` are all derived from `label` at lines
250-257. 15 of 21 categories are significant at FDR 5%; the six plotted grey are
Cellular lethal, Viable with no phenotype, Viable with phenotype, Withdrawn
Drug, Subviable and Known safety events.

Written to `data/intermediate_files/gene_pleiotropy_by_category-r1.csv`. Copying
it to `data/figure_4/gene_pleiotropy_by_category.csv` would unblock `figure_4.R`
end to end. Not done here, and Figure 4 was not re-rendered.

**Panel c is not a third irrecoverable artefact.** It is the same failure mode
as SR6's cluster table: a notebook that computes the right thing and never
persists it. The list of genuinely irrecoverable published artefacts stands at
two — the Gaussian-mixture block, which has no code anywhere, and the
`1,793 (18%)` concordance figure, which no threshold on the surviving column
reproduces. The two Figure 3 vintages are a separate matter: both exist, and the
committed figure data matches the older one, so Figure 3 is reproducible, just
not from the notebook one would reach for first.

Panel c is also **already on the resolved column** — gPS `uniqueDiseases` counts
distinct `diseaseIds` — so the trait-column change does not affect it.
