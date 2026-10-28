"""Compare computed numbers against the manuscript.

    uv run python tools/check_numbers.py

Reads `tools/expected_numbers.tsv` and every `results/*.json`, writes `REPRODUCIBILITY.md`.
Each JSON maps an id from the TSV to the value the notebook computed.

Two columns carry the manuscript side. `published` is the value originally submitted and never
changes. `expected` is the value the current `.tex` should carry, and is what the verdict compares
against. A row where the two differ is a manuscript edit that is owed, not a reproduction failure:
it is marked EDIT in the report and counted at the top.

A row's `status` column may be `pending` (compare it), `blocked` (an input this repository does
not have) or `precomputed` (a number produced upstream of this pipeline, before the validation it
starts from, and not recoverable from the released data).
"""

import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TSV = ROOT / "tools" / "expected_numbers.tsv"
RESULTS = ROOT / "results"
OUT = ROOT / "REPRODUCIBILITY.md"


def computed():
    """Merge every results JSON into one id -> value map."""
    values = {}
    for path in sorted(RESULTS.glob("*.json")):
        for key, value in json.loads(path.read_text()).items():
            values[key] = (value, path.name)
    return values


def verdict(expected, tolerance, got):
    """PASS if |got - expected| <= tolerance, else MISMATCH."""
    try:
        return "PASS" if abs(float(got) - float(expected)) <= float(tolerance) else "MISMATCH"
    except (TypeError, ValueError):
        return "PASS" if str(got) == str(expected) else "MISMATCH"


def main():
    """Write the reproducibility report."""
    rows = list(csv.DictReader(TSV.read_text().splitlines(), delimiter="\t"))
    values = computed()

    lines = ["# Reproducibility", ""]
    counts = {"PASS": 0, "MISMATCH": 0, "BLOCKED": 0, "PRECOMPUTED": 0, "PENDING": 0}
    table, owed = [], []
    for row in rows:
        got, source = values.get(row["id"], (None, ""))
        if row["status"] == "blocked":
            state = "BLOCKED"
        elif row["status"] == "precomputed":
            state = "PRECOMPUTED"
        elif got is None:
            state = "PENDING"
        else:
            state = verdict(row["expected"], row["tolerance"], got)
        counts[state] += 1
        edit = row["expected"] != row["published"]
        owed.append(row["id"]) if edit else None
        table.append(
            (state, row["id"], row["section"], row["description"], row["expected"], row["published"], got, source)
        )

    lines.append(" | ".join(f"{k} {v}" for k, v in counts.items()))
    lines.append("")
    lines.append(
        f"{len(owed)} rows where `expected` differs from `published`: the manuscript still carries the "
        "originally submitted value and owes an edit. They are marked EDIT below."
    )
    lines += [
        "",
        "| | id | section | claim | expected | published | computed | source |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = {"MISMATCH": 0, "BLOCKED": 1, "PRECOMPUTED": 2, "PENDING": 3, "PASS": 4}
    for state, rid, section, desc, exp, pub, got, source in sorted(table, key=lambda r: (order[r[0]], r[1])):
        mark = f"{state} EDIT" if exp != pub else state
        lines.append(
            f"| {mark} | {rid} | {section} | {desc} | {exp} | {pub} | {'' if got is None else got} | {source} |"
        )

    OUT.write_text("\n".join(lines) + "\n")
    print(OUT, counts, "| manuscript edits owed:", len(owed))


if __name__ == "__main__":
    main()
