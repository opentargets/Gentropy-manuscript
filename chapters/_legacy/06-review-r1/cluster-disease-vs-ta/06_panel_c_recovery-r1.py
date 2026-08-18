"""Recover Figure 4 panel c's input table.

`figure_4.R:213` reads `data/figure_4/gene_pleiotropy_by_category.csv`, which is absent from disk.
`chapters/02-analysis/05-gene-level-ps/04_gene_pleiotropy_by categories.ipynb` builds exactly that
table (`results_df`, with columns label / odds_ratio / log_odds_ratio / ci_lower / ci_upper /
log_ci_lower / log_ci_upper / p_value) but never writes it -- there is no `to_csv` in the notebook,
which is why a content grep for the filename found only the two consumers.

That notebook needs `{output_path}genes_pleiotropy`, which is also absent. It is produced by
`01_gene_level_pleiotropy.ipynb` cell 13. Cell 3 of that notebook reads
`{release_path}target_prioritisation` and cell 10 reads `/users/dc16/data/releases/25.06/...`,
neither of which exists here -- but both feed columns appended *after* the cell-9 aggregation, so
neither is needed for `geneId` and `uniqueDiseases`, the only two columns panel c consumes.

This script therefore ports, Spark-free:
  * `01_gene_level_pleiotropy.ipynb` cells 3, 6, 9 -- restricted to geneId + uniqueDiseases
  * `04_gene_pleiotropy_by categories.ipynb` cells 4-11 in full

CONTROL. Panel c of the published `figure_4_final.pdf` is vector text, so its 21 row labels carry
`total_in_category` and `pct_overlap` verbatim (figure_4.R splits notebook 04's
`f"{category} ({total}/{pct:.1f}%)"` into the "Genes" and "In set" columns). Those 42 values, plus
the 8,285-gene count printed by notebook 01 cell 12 and the 4,445 human-knockout count printed by
notebook 04 cell 6, are asserted here before anything is written.

Run from the repository root.
"""

import sys

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import statsmodels.api as sm

sys.path.insert(0, "chapters/06-review-r1/cluster-disease-vs-ta")
import cluster_lib_r1 as lib  # noqa: E402

OUT = lib.INTERMEDIATE + "gene_pleiotropy_by_category-r1.csv"

# --- notebook 04 cell 7: source label mapping --------------------------------
CATEGORY_MAPPING = {
    "all_diseases": "GWAS", "cancer_ChEMBL": "Cancer ChEMBL", "gene_burden": "Gene-based analysis",
    "omim": "OMIM", "cancer_driver_gene": "Cancer Driver (COSMIC)", "withdrawn_drug": "Withdrawn Drug",
    "non_cancer_ChEMBL": "Non-Cancer ChEMBL", "essential_gene": "Essential Gene (DepMap)",
    "gwas_eQTL": "GWAS with eQTL evidence", "gwas_with_pav": "GWAS with PAV evidence",
    "liable_target": "Known safety events", "orphanet": "Orphanet", "ChEMBL": "ChEMBL",
    "dd_related": "DD panel (gene2phenotype)", "pharmacogene": "Pharmacogenetics - Toxicity",
    "trial_safety_concern": "Trial Safety", "mouse_ko_mortality": "Mouse KO Mortality",
    "All genes": "All protein-coding genes", "fusil_CL": "Cellular lethal (FUSIL)",
    "fusil_DL": "Developmental lethal (FUSIL)", "fusil_SV": "Subviable (FUSIL)",
    "fusil_VP": "Viable with phenotype (FUSIL)", "fusil_VN": "Viable with no phenotype (FUSIL)",
    "lof_constr_Q4": "Q4 LoF constraint", "lof_constr_Q3": "Q3 LoF constraint",
    "lof_constr_Q2": "Q2 LoF constraint", "lof_constr_Q1": "Q1 LoF constraint",
    "non_essential": "Non-essential Gene (DepMap)", "distant_ortholog": "Drosophila distant orthologs",
    "human_ko": "Human Knockout",
}
CATEGORIES_TO_INCLUDE = [
    "Gene-based analysis", "OMIM", "Cancer Driver (COSMIC)", "Withdrawn Drug",
    "Essential Gene (DepMap)", "Known safety events", "Orphanet", "ChEMBL",
    "DD panel (gene2phenotype)", "Trial Safety", "Mouse KO Mortality", "Cellular lethal (FUSIL)",
    "Developmental lethal (FUSIL)", "Subviable (FUSIL)", "Viable with phenotype (FUSIL)",
    "Viable with no phenotype (FUSIL)", "Q4 LoF constraint", "Q1 LoF constraint",
    "Non-essential Gene (DepMap)", "Drosophila distant orthologs", "Human Knockout",
]

# --- published panel c, read off figure_4_final.pdf with pdftotext -----------
# figure_4.R relabels the categories for display; mapped back to notebook 04's names here.
PUBLISHED = {
    "Cancer Driver (COSMIC)": (368, 68.8), "Gene-based analysis": (557, 60.9),
    "DD panel (gene2phenotype)": (861, 57.4), "Trial Safety": (440, 55.7),
    "Mouse KO Mortality": (1022, 57.8), "OMIM": (4182, 51.6), "Q4 LoF constraint": (4520, 56.7),
    "Orphanet": (3814, 50.8), "ChEMBL": (1137, 54.9), "Known safety events": (214, 56.5),
    "Developmental lethal (FUSIL)": (785, 50.8), "Subviable (FUSIL)": (419, 52.7),
    "Withdrawn Drug": (166, 54.8), "Viable with phenotype (FUSIL)": (1862, 50.7),
    "Viable with no phenotype (FUSIL)": (321, 45.2), "Human Knockout": (4445, 41.3),
    "Cellular lethal (FUSIL)": (415, 39.0), "Non-essential Gene (DepMap)": (766, 31.5),
    "Essential Gene (DepMap)": (1489, 35.3), "Q1 LoF constraint": (4526, 33.2),
    "Drosophila distant orthologs": (830, 41.9),
}

# --- notebook 01 cells 3, 6, 9: gPS per gene ---------------------------------
studies = ds.dataset(lib.INTERMEDIATE + "gwas_w_therapeutic_areas", format="parquet").to_table(
    columns=["studyId", "diseaseIds", "measurement", "binaryLessCases"]
).to_pandas()
studies = studies[(~studies.measurement.astype(bool)) & (studies.binaryLessCases.astype(bool))]
qualified = set(ds.dataset(lib.INTERMEDIATE + "qualifying_gwas_studies", format="parquet")
                .to_table(columns=["studyId"]).to_pandas().studyId)
studies = studies[studies.studyId.isin(qualified)]
print(f"studies after ~measurement & binaryLessCases & qualifying semi-join: {len(studies):,}")

l2g = ds.dataset(lib.INTERMEDIATE + "list_of_prioritised_genes_per_CS.parquet",
                 format="parquet").to_table(columns=["studyLocusId", "geneId"]).to_pandas()
cred = ds.dataset(lib.INTERMEDIATE + "qualifying_credible_sets", format="parquet").to_table(
    columns=["studyLocusId", "studyId"]).to_pandas()

signif = l2g.merge(cred, on="studyLocusId", how="inner").merge(
    studies[["studyId", "diseaseIds"]], on="studyId", how="inner")
print(f"l2g_signif rows: {len(signif):,}")

gps = (signif.groupby("geneId")["diseaseIds"]
       .apply(lambda col: len({d for arr in col if arr is not None for d in arr}))
       .rename("uniqueDiseases").reset_index())
print(f"genes with a gPS: {len(gps):,}   (notebook 01 cell 12 printed 8285)")
assert len(gps) == 8285, f"gene count {len(gps)} != 8285"
print("PASS  gene count")

# --- notebook 04 cells 4-6: human knockout ----------------------------------
hko = ds.dataset("data/41586_2024_7556_MOESM8_ESM.csv", format="csv").to_table(
    columns=["TranscriptId"]).to_pandas()
target = ds.dataset(lib.RELEASE + "target", format="parquet").to_table(
    columns={"targetId": ds.field("id"),
             "transcript": pc.struct_field(ds.field("canonicalTranscript"), "id")}).to_pandas()
hko_processed = target.merge(hko, left_on="transcript", right_on="TranscriptId", how="inner")[["targetId"]]
hko_processed = hko_processed.assign(source="human_ko").drop_duplicates()
print(f"\nhuman-knockout genes: {len(hko_processed):,}   (notebook 04 cell 6 printed 4445)")
assert len(hko_processed) == 4445, f"HKO count {len(hko_processed)} != 4445"
print("PASS  human-knockout count")

# --- notebook 04 cells 9-10: categories and totals --------------------------
cats = ds.dataset(lib.INTERMEDIATE + "list_of_genes_32_categories", format="parquet").to_table(
    columns=["targetId", "source"]).to_pandas()
cats = pd.concat([cats, hko_processed], ignore_index=True)
cats["source"] = cats.source.replace(CATEGORY_MAPPING)
cats = cats[cats.source.isin(CATEGORIES_TO_INCLUDE)].drop_duplicates(["targetId", "source"])
category_totals = cats.groupby("source").targetId.nunique().to_dict()

merged = gps.merge(cats, left_on="geneId", right_on="targetId", how="inner")
df_pd = gps[gps.geneId.isin(set(merged.geneId))].copy()
print(f"\ngenes in >=1 included category and having a gPS: {len(df_pd):,}")

# --- notebook 04 cell 11: logistic regression per category ------------------
df_pd["log2_uniqueDiseases"] = np.log2(df_pd.uniqueDiseases)
results = []
for category in sorted(cats.source.unique()):
    in_cat = set(merged.loc[merged.source == category, "geneId"])
    y = df_pd.geneId.isin(in_cat).astype(int)
    n_in_cat = int(y.sum())
    X = sm.add_constant(df_pd.log2_uniqueDiseases.astype(float))
    try:
        model = sm.Logit(y, X).fit(disp=0)
    except Exception as exc:            # notebook 04 wraps the fit in try/except
        print(f"  fit failed for {category}: {exc}")
        continue
    log_or = model.params["log2_uniqueDiseases"]
    conf = model.conf_int()
    lo, hi = conf.loc["log2_uniqueDiseases", 0], conf.loc["log2_uniqueDiseases", 1]
    total = category_totals.get(category, 0)
    pct = (n_in_cat / total * 100) if total else 0.0
    results.append({"category": category, "label": f"{category} ({total}/{pct:.1f}%)",
                    "odds_ratio": np.exp(log_or), "log_odds_ratio": log_or,
                    "ci_lower": np.exp(lo), "ci_upper": np.exp(hi),
                    "log_ci_lower": lo, "log_ci_upper": hi,
                    "p_value": model.pvalues["log2_uniqueDiseases"],
                    "n_in_category": n_in_cat, "total_in_category": total, "pct_overlap": pct})

results_df = pd.DataFrame(results).sort_values("log_odds_ratio", ascending=True)

# --- CONTROL against the published panel c ---------------------------------
print(f"\ncontrol: {len(results_df)} categories vs {len(PUBLISHED)} published rows")
ok = len(results_df) == len(PUBLISHED)
print(f"  {'PASS' if ok else 'FAIL'}  category count")
rows = []
for r in results_df.itertuples():
    want = PUBLISHED.get(r.category)
    if want is None:
        rows.append((r.category, r.total_in_category, "-", r.pct_overlap, "-", "NOT IN PUBLISHED"))
        ok = False
        continue
    tot_ok = r.total_in_category == want[0]
    pct_ok = abs(round(r.pct_overlap, 1) - want[1]) < 0.05
    ok &= tot_ok and pct_ok
    rows.append((r.category, r.total_in_category, want[0], round(r.pct_overlap, 1), want[1],
                 "PASS" if tot_ok and pct_ok else "FAIL"))
diff = pd.DataFrame(rows, columns=["category", "genes", "published_genes",
                                   "in_set_pct", "published_pct", "verdict"])
print(diff.to_string(index=False))
print(f"\n{'CONTROL PASSED' if ok else 'CONTROL FAILED'}")
if not ok:
    raise SystemExit("panel c control failed -- not writing the recovered table")

results_df.drop(columns=["n_in_category", "total_in_category", "pct_overlap"]).to_csv(OUT, index=False)
print(f"\nwrote {OUT}  {results_df.shape}")
print(results_df[["category", "log_odds_ratio", "log_ci_lower", "log_ci_upper", "p_value"]].to_string(index=False))
