# How stable are gPS and gps_TA to perturbation of the disease list?

The published pleiotropy metrics are counts over the **1,394 disease terms that
happen to have a qualifying GWAS**. That list is an accident of what has been
studied, and it will grow. This folder asks what happens if 20% of it had been
missing: does the manuscript's pattern hold, and how likely is it to look
different in future releases?

100 replicates. Each drops a uniformly random 20% of disease terms (**1,394 →
1,115**), recomputes gPS and gps_TA over the survivors, and re-runs three things
— rank agreement with the full-data metric, the non-linear relationship with
drug approval, and the high-versus-low pleiotropy enrichment.

## Headline

**The pattern is qualitatively immovable and quantitatively soft.**

| Conclusion                                                              | Replicates preserving it |
| ----------------------------------------------------------------------- | ------------------------ |
| non-linearity (quadratic term) significant at P < 0.001 — gPS           | **100 / 100**            |
| non-linearity significant at P < 0.001 — gps_TA                         | **100 / 100**            |
| ceiling in the right direction (ratio > 1), frozen cuts — gPS           | **100 / 100**            |
| ceiling in the right direction, frozen cuts — gps_TA                    | **100 / 100**            |
| ceiling still significant at P < 0.05, frozen cuts — gPS                | **98 / 100**             |
| ceiling still significant at P < 0.05, frozen cuts — gps_TA             | **34 / 100**             |
| ceiling still significant, cuts re-derived per replicate — gPS / gps_TA | 65 / 100 and 71 / 100    |

Worst quadratic P value across all 200 fits: **2.6 × 10⁻⁶** (gPS) and **5.2 ×
10⁻⁷** (gps_TA). The inverted-U is not an artefact of which diseases happen to
be in the catalogue.

Rank agreement with the full-data metric is high but not near-perfect: Spearman
**median 0.950** (gPS, range 0.927–0.971) and **0.936** (gps*TA, 0.897–0.963). A
20% disease drop reshuffles individual gene rankings by a few percent, so
\_gene-level* statements are less durable than the population-level pattern.

## Decisions taken before running anything

| Decision                         | Choice                                                                                | Why                                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| gps_TA on a cropped disease list | **one TA per disease**, `therapy_area_hierarchy` priority (first match, else `other`) | the only decomposable rule faithful to the published column                                               |
| which diseases are dropped       | **uniform at random**, 20%                                                            | cleanest null; matches "we happen not to have GWAS for these"                                             |
| genetic support                  | **held fixed**                                                                        | `score_all` is a max over propagated associations and cannot be decomposed by disease from the pair table |

### The gps_TA decomposition, and what it costs

The published `uniqueTherapeuticAreas` is a **study**-level union — each
contributing study contributes the areas spanned by _all_ of its `diseaseIds`.
That is not decomposable by disease: a disease inherits areas from its
study-mates, so dropping it would leave its borrowed areas attached to the
survivors and the count would fail to fall when it should. Two decomposable
alternatives, measured against the published column over all 8,285 genes:

| Rule                                          | Exact match | Mean | Max |
| --------------------------------------------- | ----------- | ---- | --- |
| **one TA per disease, hierarchy first-match** | **86.3%**   | 2.42 | 20  |
| every top-level area a disease descends from  | 26.8%       | 4.06 | 22  |
| _published_                                   | —           | 2.53 | 21  |

The multi-area rule inflates by 60%, because **627 of the 1,394** disease terms
sit under more than one top-level area. The hierarchy rule is used here.

**Consequence that must be carried into any quotation of these numbers:** the
100% baseline is recomputed with the same rule, so subsampling is never
confounded with the definitional difference — but the gps_TA baseline in this
folder is therefore **not** the number in `../effective-independent-traits/`.
Here the full-data gps_TA ceiling is **ratio 1.545, P = 0.072**; on the
published column it is 1.90, P = 0.0094. Comparisons must stay inside one
folder. gPS needs no such care: it is a plain count of distinct disease terms
and the baseline reproduces the published `uniqueDiseases` for **8,285 of 8,285
genes**.

## Notebook

| Notebook                      | Needs Spark | Runtime | What it does                                              |
| ----------------------------- | ----------- | ------- | --------------------------------------------------------- |
| `01_subsample_diseases.ipynb` | no          | ~3 min  | builds the TA map, the 100% baseline, then 100 replicates |

Imports `or_rs` and `support_mask` from
`../or10-optimism-validation/or10_stats.py`, so the support definition and the
Fisher/Woolf arithmetic are identical to every other analysis in `06-review-r1`.

```bash
cd chapters/06-review-r1/disease-subsampling
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 01_subsample_diseases.ipynb
```

## Baseline, 100% of diseases

Both metrics run through exactly the same code path as the replicates.

| Metric   | quadratic LR | fitted peak | decay point | derived cuts | OR low | OR high | ratio     | P          |
| -------- | ------------ | ----------- | ----------- | ------------ | ------ | ------- | --------- | ---------- |
| `gps`    | **54.17**    | 3.69        | 21.04       | ≤ 4 / ≥ 22   | 4.873  | 2.627   | **1.855** | **0.0098** |
| `gps_TA` | **57.44**    | 1.906       | 7.44        | ≤ 2 / ≥ 8    | 4.209  | 2.725   | 1.545     | 0.072      |

Cuts come from the same peak / decay-point rule as
`../effective-independent-traits/02_drug_targets.ipynb`: low = M ≤ round(peak),
high = M ≥ ⌈decay⌉. All-GWAS baseline enrichment on the full 37,377-pair table:
**OR 3.6186**, matching the published 3.62.

## What each replicate does to the metrics

Dropping 20% of disease terms costs:

- **gPS**: mean 4.449 → **3.858** (−13.3%), median max 148 → 117
- **gps_TA**: mean 2.423 → **2.280** (−5.9%), median max 20 → 19
- **640 of 8,285 genes (7.7%)** lose every disease association and drop out of
  the gene set entirely (mean over replicates, from
  `subsample_disease_verdict-r1.csv`; median 628.5)

gps*TA shrinks less than gPS, as it must — losing one of several diseases in the
same therapeutic area costs a disease but not an area. That is also why gps_TA's
\_cut* is far more stable (below).

## Stability of each quantity (`subsample_disease_summary-r1.csv`)

Median and 2.5–97.5 percentile interval over 100 replicates, against the
baseline:

| Metric   | Quantity            | Baseline | Median    | 2.5–97.5%   |
| -------- | ------------------- | -------- | --------- | ----------- |
| `gps`    | Spearman vs full    | 1.0      | **0.950** | 0.930–0.967 |
| `gps`    | quadratic LR        | 54.17    | 50.77     | 28.72–67.33 |
| `gps`    | fitted peak         | 3.695    | 3.204     | 2.837–3.500 |
| `gps`    | ratio, frozen cuts  | 1.855    | **2.472** | 1.804–3.040 |
| `gps`    | ratio, derived cuts | 1.855    | **1.640** | 1.323–2.208 |
| `gps_TA` | Spearman vs full    | 1.0      | **0.936** | 0.902–0.958 |
| `gps_TA` | quadratic LR        | 57.44    | 56.61     | 29.30–79.14 |
| `gps_TA` | fitted peak         | 1.906    | 1.760     | 1.513–1.903 |
| `gps_TA` | ratio, frozen cuts  | 1.545    | **1.556** | 1.303–1.862 |
| `gps_TA` | ratio, derived cuts | 1.545    | **1.666** | 1.421–1.975 |

### Read the frozen-cut gPS row carefully — it is an artefact, not a strengthening

gPS's frozen ratio _rises_ from 1.855 to a median of 2.472. That is not the
effect getting stronger; it is the frozen cut biting into a shrunken scale. With
20% of diseases gone, far fewer genes clear the frozen high cut of ≥ 22, so the
high cell collapses from 150 pairs (39 approved) at baseline to a median of **80
pairs (16 approved)** — a smaller, more extreme group, and a wider, noisier
ratio. The **derived-cut** column is the fair comparison, and it sits slightly
_below_ baseline at 1.640. The same mechanism explains why the fitted peak
drifts down (3.69 → 3.20): the peak is measured on a metric whose whole scale
has contracted 13%.

gps_TA has no such problem — its scale contracts only 6%, its frozen high cell
holds a median 120 pairs (32 approved), and its frozen and derived ratios agree
with baseline to within 8%.

## Would the threshold itself survive? (`subsample_disease_cut_frequency-r1.csv`)

The derivation rule re-run inside each replicate:

- **gps_TA is stable**: it picks ≤ 2 / ≥ 7 in 61 replicates and ≤ 2 / ≥ 8 in 22
  — **83 of 100 land within one unit of the frozen ≥ 8**, and 100 of 100 within
  three. The low cut is ≤ 2 in 98 of 100.
- **gPS is not**: the median derived high cut is **17**, not 22, and only **2 of
  100** land within one unit of the frozen cut (20 of 100 within three). The low
  cut moves from 4 to 3 in 97 of 100 replicates.

Both shifts are the expected consequence of a contracted scale rather than
instability of the underlying shape — but they say plainly that **a numeric gPS
threshold quoted from today's disease list should not be expected to survive to
the next release**, whereas a therapeutic-area threshold probably will.

## What this means for the manuscript

1. **The qualitative claims are safe.** The inverted-U and the direction of the
   pleiotropy ceiling survive every one of 100 perturbations, for both metrics,
   with worst-case P values around 10⁻⁶ for the non-linearity. A future release
   adding diseases will not overturn them.
2. **The gPS ceiling's significance is robust; gps_TA's is not.** At frozen cuts
   gPS keeps P < 0.05 in 98 of 100 replicates against gps_TA's 34 of 100. Under
   this folder's TA definition the full-data gps_TA contrast is already only P =
   0.072, so that fragility is inherited from the baseline rather than created
   by subsampling.
3. **Specific numbers are softer than the pattern.** Rank correlation of
   0.94–0.95, a 13% contraction of the gPS scale, and a derived high cut moving
   22 → 17 all say that exact thresholds and per-gene pleiotropy values are
   properties of the current catalogue. Ratios and directions travel; cut points
   do not.

## Limitations, stated

- **Genetic support is held fixed.** A fuller counterfactual would also remove
  the support that came only through a dropped disease, which would attenuate
  the enrichment further. `score_all` is a max over propagated associations and
  cannot be decomposed by disease from the pair table, so doing this properly
  needs a Spark rebuild per replicate. Every number here is therefore an
  estimate of the pleiotropy metric's own stability, not of the whole
  pipeline's.
- **Uniform dropping is the cleanest null, not the most realistic scenario.** A
  real discovery frontier would remove less-studied diseases preferentially,
  which would bite harder on the low-gPS tail. Not run.
- **gps_TA uses a different definition here** than in
  `../effective-independent-traits/` — see above. The 86.3% agreement means
  about one gene in seven has a different TA count under the two rules.
- The replicate P values are not corrected for multiplicity; they are read as a
  distribution over perturbations, not as 100 independent hypothesis tests.
- Only the all-GWAS support definition was used. The PAV-stratified version was
  not re-run.

## Exports (all in `data/intermediate_files/`)

| File                                     | Contents                                                                   |
| ---------------------------------------- | -------------------------------------------------------------------------- |
| `subsample_disease_replicates-r1.csv`    | one row per replicate × metric — every statistic computed                  |
| `subsample_disease_summary-r1.csv`       | median and 2.5–97.5 percentile interval per quantity, against the baseline |
| `subsample_disease_verdict-r1.csv`       | how many of the 100 replicates preserve each conclusion                    |
| `subsample_disease_cut_frequency-r1.csv` | which low/high cuts the derivation rule picks across replicates            |
