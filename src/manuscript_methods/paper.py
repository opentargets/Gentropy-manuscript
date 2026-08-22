"""Paths, therapeutic-area hierarchy and result reporting for the manuscript pipeline.

Every refactored notebook imports from here so there is one definition of each.
Notebooks are executed from the repository root.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "data" / "25.06" / "output"
BASELINE = ROOT / "data" / "intermediate_files"
DERIVED = ROOT / "data" / "intermediate_files_refactor"
RESULTS = ROOT / "results"

# Order matters: the first ancestor that matches wins, unmatched diseases become "other".
# This is the order published as Supplementary Table 9, and the one that reproduces the
# published gene-level therapeutic-area counts (4,743 genes in more than one area, mean 2.53,
# max 21). The pre-refactor `04_qualifying_dataset_generation.ipynb` placed
# `genetic, familial or congenital disease` last but one instead, which gives 4,662 / 2.43 / 20.
THERAPEUTIC_AREAS = {
    "EFO_0001444": "measurement",
    "MONDO_0045024": "cancer or benign tumor",
    "OTAR_0000018": "genetic, familial or congenital disease",
    "EFO_0005741": "infectious disease",
    "OTAR_0000009": "injury, poisoning or other complication",
    "OTAR_0000014": "pregnancy or perinatal disease",
    "MONDO_0024458": "disorder of visual system",
    "EFO_0000319": "cardiovascular disease",
    "EFO_0009605": "pancreas disease",
    "EFO_0010282": "gastrointestinal disease",
    "OTAR_0000017": "reproductive system or breast disease",
    "EFO_0010285": "integumentary system disease",
    "EFO_0001379": "endocrine system disease",
    "OTAR_0000010": "respiratory or thoracic disease",
    "EFO_0009690": "urinary system disease",
    "OTAR_0000006": "musculoskeletal or connective tissue disease",
    "MONDO_0021205": "disorder of ear",
    "EFO_0000540": "immune system disease",
    "EFO_0005803": "hematologic disease",
    "EFO_0000618": "nervous system disease",
    "MONDO_0002025": "psychiatric disorder",
    "OTAR_0000020": "nutritional or metabolic disease",
    "EFO_0003765": "sign or symptom",
}

MEASUREMENT = "EFO_0001444"

# The pre-refactor pipeline used a second ordering, in which `genetic, familial or congenital
# disease` comes last but one. The variant and cluster analyses were built on it, and the
# gene-level analysis on the published order above. Both are kept so the manuscript's numbers
# reproduce exactly; unify on the published order once reproduction is established (GAPS.md).
THERAPEUTIC_AREAS_LEGACY = {
    "EFO_0001444": "measurement",
    "MONDO_0045024": "cancer or benign tumor",
    "EFO_0005741": "infectious disease",
    "OTAR_0000009": "injury, poisoning or other complication",
    "OTAR_0000014": "pregnancy or perinatal disease",
    "MONDO_0024458": "disorder of visual system",
    "EFO_0000319": "cardiovascular disease",
    "EFO_0009605": "pancreas disease",
    "EFO_0000540": "immune system disease",
    "EFO_0010282": "gastrointestinal disease",
    "OTAR_0000017": "reproductive system or breast disease",
    "EFO_0010285": "integumentary system disease",
    "EFO_0001379": "endocrine system disease",
    "OTAR_0000010": "respiratory or thoracic disease",
    "EFO_0009690": "urinary system disease",
    "OTAR_0000006": "musculoskeletal or connective tissue disease",
    "MONDO_0021205": "disorder of ear",
    "EFO_0005803": "hematologic disease",
    "EFO_0000618": "nervous system disease",
    "MONDO_0002025": "psychiatric disorder",
    "OTAR_0000020": "nutritional or metabolic disease",
    "OTAR_0000018": "genetic, familial or congenital disease",
    "EFO_0003765": "sign or symptom",
}

# One-hot column name per therapeutic area in the study and gene tables. Measurement is
# deliberately absent: it is carried as its own boolean flag, not as a therapeutic area.
TA_COLUMNS = {
    "MONDO_0045024": "cancerOrBenignTumor",
    "EFO_0005741": "infectiousDisease",
    "OTAR_0000014": "pregnancyOrPerinatalDisease",
    "MONDO_0024458": "disorderOfVisualSystem",
    "EFO_0000319": "cardiovascularDisease",
    "EFO_0009605": "pancreasDisease",
    "EFO_0010282": "gastrointestinalDisease",
    "OTAR_0000017": "reproductiveSystemOrBreastDisease",
    "EFO_0010285": "integumentarySystemDisease",
    "EFO_0001379": "endocrineSystemDisease",
    "OTAR_0000010": "respiratoryOrThoracicDisease",
    "EFO_0009690": "urinarySystemDisease",
    "OTAR_0000006": "musculoskeletalOrConnectiveTissueDisease",
    "MONDO_0021205": "disorderOfEar",
    "EFO_0000540": "immuneSystemDisease",
    "EFO_0005803": "hematologicDisease",
    "EFO_0000618": "nervousSystemDisease",
    "MONDO_0002025": "psychiatricDisorder",
    "OTAR_0000020": "nutritionalOrMetabolicDisease",
    "OTAR_0000018": "geneticFamilialOrCongenitalDisease",
    "OTAR_0000009": "injuryPoisoningOrOtherComplication",
    "EFO_0003765": "signOrSymptom",
    "other": "other",
}


def release(name: str) -> str:
    """Path to a dataset in the downloaded Open Targets release."""
    return str(RELEASE / name)


def derived(name: str) -> str:
    """Path to a dataset written by this pipeline, creating the parent directory."""
    DERIVED.mkdir(parents=True, exist_ok=True)
    return str(DERIVED / name)


def baseline(name: str) -> str:
    """Path to the equivalent pre-refactor dataset, used only for cross-checks."""
    return str(BASELINE / name)


def first_therapeutic_area(ancestors) -> str:
    """First therapeutic area among a disease's ontology ancestors, else "other"."""
    if not ancestors:
        return "other"
    seen = set(ancestors)
    for area in THERAPEUTIC_AREAS:
        if area in seen:
            return area
    return "other"


def save_results(name: str, values: dict) -> str:
    """Write the numbers a notebook computed to results/<name>.json.

    numpy scalars arrive whenever a value is read out of a DataFrame, and `json` cannot
    serialise them, so they are unwrapped here rather than at every call site.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    plain = {key: value.item() if hasattr(value, "item") else value for key, value in values.items()}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(plain, indent=2, sort_keys=True) + "\n")
    return str(path)
