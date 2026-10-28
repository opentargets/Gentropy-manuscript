"""Rebuild the Supplementary Table 13 sheet from the CSVs in manual/refreshed/.

The sheet is hand-laid-out, so this keeps its layout: a title row, a description row, one note per
column, then four subtables, each with a heading row, a blank row and a header. Only the numbers,
the therapeutic-area rows and two corrected notes change.

    uv run --with openpyxl python chapters/06-supplementary-tables/manual/build_st8.py

Writes manual/refreshed/ST13_-_subgroup_analysis.xlsx. Copying it over
supplementary_tables/sheets/ST13_-_subgroup_analysis.xlsx in the manuscript repository is a
deliberate, separate step.

The script name and the hand-laid-out `manual/ST8_-_subgroup_analysis.xlsx` keep this repository's
own number for the table (8); the root README maps it to the printed number (13).
"""

from pathlib import Path

import openpyxl
import pandas as pd

from manuscript_methods import paper

HERE = Path(__file__).resolve().parent
REFRESHED = HERE / "refreshed"

# The published class order, kept so the sheet reads the same. Classes with no pairs stay as zero rows.
CLASS_ORDER = [
    "Enzyme",
    "Other nuclear protein",
    "Adhesion",
    "Surface antigen",
    "Membrane receptor",
    "Epigenetic regulator",
    "Other cytosolic protein",
    "Other membrane protein",
    "Ion channel",
    "Auxiliary transport protein",
    "Transcription factor",
    "Unclassified protein",
    "Transporter",
    "Secreted protein",
    "Structural protein",
]

TITLE = "Supplementary Table 13 | Distribution of gene–disease pairs by therapeutic area and target class."
DESCRIPTION = (
    "This table contains four subtables: (1) all prioritised gene–disease pairs by target class; "
    "(2) all prioritised gene–disease pairs by therapeutic area; (3) PAV + 2–5 therapeutic areas "
    "subgroup by target class; (4) PAV + 2–5 therapeutic areas subgroup by therapeutic area. "
    "Each disease is assigned to exactly one therapeutic area (Supplementary Table 14), so subtables 2 "
    "and 4 sum to the distinct pairs. A target with several ChEMBL level-1 classes is counted in each, "
    "so subtables 1 and 3 can sum to more than the distinct pairs; targets with no ChEMBL class are "
    "not included."
)
NOTES = [
    ("Target Class (l1_class)", "ChEMBL level-1 target class category (e.g., Enzyme, Membrane receptor, Ion channel, Transcription factor)."),
    ("Therapeutic Area", "Therapeutic area of the disease, assigned as in Supplementary Table 14; 'other (no area root)' collects diseases under none of the areas."),
    ("Ontology ID", "EFO/MONDO/OTAR ontology identifier for the therapeutic area; empty for 'other (no area root)'."),
    ("Number of Pairs (N)", "Total number of gene–disease association pairs in this subgroup."),
    ("Unique Diseases", "Number of unique diseases represented in this subgroup."),
    ("Unique Targets", "Number of unique gene targets represented in this subgroup."),
]
HEADINGS = [
    "Table 1: Distribution of all nominated genetic evidence by target class:",
    "Table 2: Distribution of all nominated genetic evidence by therapeutic area",
    "Table 3: Distribution of the best subgroup of genetic evidence (PAV + number of TA from 2 to 5) by target class.",
    "Table 4: Distribution of the best subgroup of genetic evidence (PAV + number of TA from 2 to 5) by therapeutic area.",
]
COUNTS = ["Number of Pairs (N)", "Unique Diseases", "Unique Targets"]


def class_rows(name):
    table = pd.read_csv(REFRESHED / f"{name}.csv").set_index("Target Class (l1_class)")
    unknown = set(table.index) - set(CLASS_ORDER)
    assert not unknown, f"classes missing from CLASS_ORDER: {unknown}"
    table = table.reindex(CLASS_ORDER).fillna(0).astype(int)
    return [["Target Class (l1_class)", *COUNTS]] + [[label, *map(int, row)] for label, row in table[COUNTS].iterrows()]


def area_rows(name):
    table = pd.read_csv(REFRESHED / f"{name}.csv").set_index("Ontology ID")
    order = [area for area in [*paper.THERAPEUTIC_AREAS, "other"] if area in table.index]
    assert set(order) == set(table.index), "an area in the CSV is not in the Supplementary Table 14 hierarchy"
    rows = [["Therapeutic Area", "Ontology ID", *COUNTS]]
    for area in order:
        label = table.loc[area, "Therapeutic Area"]
        label = label if area == "other" else label[0].upper() + label[1:]
        rows.append([label, None if area == "other" else area, *map(int, table.loc[area, COUNTS])])
    return rows


def main():
    blocks = [
        class_rows("ST13_table1_all_nominated_by_target_class"),
        area_rows("ST13_table2_all_nominated_by_therapeutic_area"),
        class_rows("ST13_table3_best_subgroup_by_target_class"),
        area_rows("ST13_table4_best_subgroup_by_therapeutic_area"),
    ]
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "ST13 - subgroup analysis"
    sheet.append([TITLE])
    sheet.append([])
    sheet.append([DESCRIPTION])
    sheet.append([])
    for column, note in NOTES:
        sheet.append([column, note])
    for heading, rows in zip(HEADINGS, blocks):
        sheet.append([])
        sheet.append([heading])
        sheet.append([])
        for row in rows:
            sheet.append(row)
    out = REFRESHED / "ST13_-_subgroup_analysis.xlsx"
    workbook.save(out)
    for heading, rows in zip(HEADINGS, blocks):
        print(f"{heading.split(':')[0]}: {len(rows) - 1} rows, {sum(row[-3] for row in rows[1:]):,} pairs")
    print(out)


if __name__ == "__main__":
    main()
