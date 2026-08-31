"""The L2G gold standard, the prioritisation rules scored against it, and their 2x2 tables.

Used by Supplementary Results 3 and 4 and by Supplementary Table 12. The gold standard and the
held-out split live in `data/l2g_training_set/`, which is not yet fetched by the download
notebook — see GAPS.md.
"""

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from manuscript_methods import paper

GOLD_STANDARD = paper.ROOT / "data" / "l2g_training_set" / "20250625_gentropy_paper_v1"
TEST_SPLIT = paper.ROOT / "data" / "l2g_training_set" / "test_v3.parquet"
# The manuscript's own L2G model, not the release's; the same predictions every prioritisation in
# `01-data-preparation/05_l2g_prioritised_genes.ipynb` is built from.
PREDICTIONS = paper.ROOT / "data" / "25.06" / "irene_1208_l2g_predictions"

# The thresholds 05_l2g_prioritised_genes.ipynb uses for the same evidence flags.
CLPP, H4, VEP_PAV = 0.01, 0.8, 0.66

FEATURE_COLUMNS = [
    "studyLocusId",
    "geneId",
    "eQtlColocClppMaximum",
    "eQtlColocH4Maximum",
    "pQtlColocClppMaximum",
    "pQtlColocH4Maximum",
    "vepMaximum",
    "distanceSentinelTssNeighbourhood",
]


def labelled_gold_standard() -> pd.DataFrame:
    """The gold standard with its L2G score, evidence features and held-out flag attached."""
    gold = pads.dataset(str(GOLD_STANDARD), format="parquet").to_table().to_pandas()
    gold["positive"] = (gold["goldStandardSet"] == "positive").astype(int)

    scores = (
        pads.dataset(str(PREDICTIONS), format="parquet")
        .to_table(columns=["studyLocusId", "geneId", "score"])
        .to_pandas()
    )
    features = pads.dataset(str(paper.release("l2g_feature_matrix")), format="parquet").to_table(
        columns=FEATURE_COLUMNS
    ).to_pandas()
    held_out = pd.read_parquet(TEST_SPLIT, columns=["studyLocusId", "geneId"]).assign(heldOut=True)

    labelled = (
        gold.merge(scores, on=["studyLocusId", "geneId"], how="left")
        .merge(features, on=["studyLocusId", "geneId"], how="left")
        .merge(held_out, on=["studyLocusId", "geneId"], how="left")
    )
    if len(labelled) != len(gold):
        raise ValueError(f"the joins changed the row count: {len(gold)} -> {len(labelled)}")
    labelled["heldOut"] = labelled["heldOut"].fillna(False).astype(bool)
    labelled["score"] = labelled["score"].fillna(0)
    return labelled


def evidence_masks(labelled: pd.DataFrame, combined: pd.Series | None = None) -> dict:
    """One boolean mask per prioritisation rule, in the order the published tables list them.

    Args:
        labelled: output of `labelled_gold_standard`
        combined: the manuscript's own prioritisation rule, which is materialised as
            `prioritised_genes_per_cs` rather than recomputed; omitted when not supplied
    """
    score = labelled["score"]
    masks = {
        "L2G>=0.5": score >= 0.5,
        "L2G>=0.05": score >= 0.05,
        "L2G>=0.8": score >= 0.8,
        "eQTL_coloc": (labelled["eQtlColocClppMaximum"].fillna(0) >= CLPP)
        | (labelled["eQtlColocH4Maximum"].fillna(0) >= H4),
        "pQTL_coloc": (labelled["pQtlColocClppMaximum"].fillna(0) >= CLPP)
        | (labelled["pQtlColocH4Maximum"].fillna(0) >= H4),
        "PAV": labelled["vepMaximum"].fillna(0) >= VEP_PAV,
        "Nearest to TSS": labelled["distanceSentinelTssNeighbourhood"].fillna(0) == 1,
    }
    if combined is not None:
        masks["Combined"] = combined
    return masks


def confusion(predicted, actual) -> dict:
    """The 2x2 table and the rates Supplementary Table 12 and Supplementary Results 4 report."""
    predicted, actual = np.asarray(predicted), np.asarray(actual).astype(bool)
    tp = int((predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    return {
        "Evidence": None,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Sensitivity (recall)": sensitivity,
        "Specificity (selectivity)": specificity,
        "PPV (precision)": ppv,
        "FDR": 1 - ppv if ppv == ppv else np.nan,
        "Balanced_accuracy": (sensitivity + specificity) / 2,
    }
