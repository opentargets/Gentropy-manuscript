"""Compare computed numbers against the manuscript.

    uv run python tools/check_numbers.py

Reads `tools/expected_numbers.tsv` and every `results/*.json`, writes `REPRODUCIBILITY.md`.
Each JSON maps an id from the TSV to the value the notebook computed.
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
    counts = {"PASS": 0, "MISMATCH": 0, "BLOCKED": 0, "PENDING": 0}
    table = []
    for row in rows:
        got, source = values.get(row["id"], (None, ""))
        if row["status"] == "blocked":
            state = "BLOCKED"
        elif got is None:
            state = "PENDING"
        else:
            state = verdict(row["expected"], row["tolerance"], got)
        counts[state] += 1
        table.append((state, row["id"], row["section"], row["description"], row["expected"], got, source))

    lines.append(" | ".join(f"{k} {v}" for k, v in counts.items()))
    lines += ["", "| | id | section | claim | manuscript | computed | source |", "|---|---|---|---|---|---|---|"]
    order = {"MISMATCH": 0, "BLOCKED": 1, "PENDING": 2, "PASS": 3}
    for state, rid, section, desc, exp, got, source in sorted(table, key=lambda r: (order[r[0]], r[1])):
        lines.append(f"| {state} | {rid} | {section} | {desc} | {exp} | {'' if got is None else got} | {source} |")

    OUT.write_text("\n".join(lines) + "\n")
    print(OUT, counts)


if __name__ == "__main__":
    main()
