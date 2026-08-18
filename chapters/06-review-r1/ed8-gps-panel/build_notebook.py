"""Extend the Extended Data Fig. 8 notebook with a gPS panel (referee R1-mn-14).

Patches `chapters/03-manuscript-figures/extended_figures/ed8_translation_success_by_pleiotropy.ipynb`
in place: keeps the published single-panel cells as a control, redirects their PDF so the published
`extended_figure_8.pdf` is never overwritten, and appends the two-panel (TAs, gPS) analysis.

Idempotent: appended cells carry an `r1_ed8_gps` tag and are replaced on re-run.
"""

from __future__ import annotations

import json
import pathlib

NB = (
    pathlib.Path(__file__).resolve().parents[3]
    / "chapters/03-manuscript-figures/extended_figures/ed8_translation_success_by_pleiotropy.ipynb"
)
TAG = "r1_ed8_gps"

HEADER_MD = """\
# Extended Data Figure 8 — Target-disease translation success by pleiotropy

Phase-to-phase transition probabilities (Phase I→II, II→III, III→approval) for drug
target-indication pairs, stratified by gene pleiotropy on **two** scales:

- **(a)** number of therapeutic areas — Low (1 TA), Medium (2–5 TAs), High (≥ 6 TAs)
- **(b)** gPS, the number of unique associated diseases — Low (gPS 1), Medium (gPS 2–9),
  High (gPS ≥ 10)

Panel (b) was added in response to referee comment **R1-mn-14**, which asked for the high-pleiotropy
cut point to be consistent with Fig. 5b and Fig. 5c (gPS ≥ 10). The gPS panel uses the same universe
as the published panel (`uniqueTherapeuticAreas >= 1`, 18,480 pairs) so the two panels are directly
comparable.

**Source notebook:** `chapters/02-analysis/06-target-enrichment/10-gps-in-clinical-stages.ipynb`
**Data:** `data/intermediate_files/df_for_enrichment_regression.csv`
**Response analysis:** `chapters/06-review-r1/ed8-gps-panel/`
"""

CELLS: list[tuple[str, str]] = []


def add(cell_type: str, source: str) -> None:
    """Queue a tagged cell for the appended section."""
    CELLS.append((cell_type, source))


add(
    "markdown",
    """\
## R1-mn-14 — two-panel version: therapeutic areas and gPS

Everything below re-derives the published panel through a single parameterised code path and then
applies the identical method to a gPS grouping. The published cells above are kept as a control and
their figure is written to `extended_figure_8_control_ta_only-r1.pdf`; the published
`extended_figure_8.pdf` is not touched.

Method, identical for both panels:

- universe: `uniqueTherapeuticAreas >= 1` (18,480 pairs) — held fixed across panels
- reaching phase *n* means `maxClinicalPhase >= n`; each transition rate is the share of pairs at the
  start phase that reach the end phase
- Wilson 95% intervals (`proportion_confint(..., method="wilson")`)
- pairwise two-sided proportions z-tests within each transition (Medium vs Low, High vs Low,
  High vs Medium) — nine tests, Benjamini–Hochberg across all nine
""",
)

add(
    "code",
    '''\
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

UNIVERSE = df[df["uniqueTherapeuticAreas"] >= 1].copy()
assert len(UNIVERSE) == 18480, len(UNIVERSE)

TRANSITIONS = [(1, 2, "Phase I\\u2192II"), (2, 3, "Phase II\\u2192III"), (3, 4, "Phase III\\u2192approval")]
COMPARISONS = [("med_vs_low", 1, 0), ("high_vs_low", 2, 0), ("high_vs_med", 2, 1)]
COMPARISON_LABELS = {
    "med_vs_low": "Medium vs Low",
    "high_vs_low": "High vs Low",
    "high_vs_med": "High vs Medium",
}

GROUPINGS = {
    "TAs": {
        "column": "uniqueTherapeuticAreas",
        "edges": [0, 1, 5, float("inf")],
        "labels": ["1 TA", "2\\u20135 TAs", "\\u22656 TAs"],
    },
    "gPS": {
        "column": "uniqueDiseases",
        "edges": [0, 1, 9, float("inf")],
        "labels": ["gPS 1", "gPS 2\\u20139", "gPS \\u226510"],
    },
}


def assign_bins(data, spec):
    """Cut a pleiotropy column into Low/Medium/High bins.

    Args:
        data: pair-level table restricted to the analysis universe
        spec: one entry of GROUPINGS, giving the column, bin edges and labels

    Returns:
        Categorical Series aligned to data
    """
    return pd.cut(data[spec["column"]], spec["edges"], labels=spec["labels"])


def transition_table(data, spec):
    """Per group per transition: n at start, n reaching, rate and Wilson 95% CI.

    Args:
        data: pair-level table restricted to the analysis universe
        spec: one entry of GROUPINGS

    Returns:
        DataFrame with one row per (group, transition)
    """
    binned = assign_bins(data, spec)
    rows = []
    for tier, label in zip(["Low", "Medium", "High"], spec["labels"], strict=True):
        subset = data[binned == label]
        for start_phase, end_phase, name in TRANSITIONS:
            n_start = int((subset["maxClinicalPhase"] >= start_phase).sum())
            n_reach = int((subset["maxClinicalPhase"] >= end_phase).sum())
            ci_low, ci_high = proportion_confint(n_reach, n_start, alpha=0.05, method="wilson")
            rows.append(
                {
                    "grouping": spec["name"],
                    "tier": tier,
                    "group": label,
                    "group_n": int((binned == label).sum()),
                    "transition": name,
                    "n_at_start": n_start,
                    "n_reaching": n_reach,
                    "rate": n_reach / n_start,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def pairwise_table(table, spec):
    """Nine pairwise proportion tests with BH correction across all nine.

    Both BH variants are reported: `p_adj_bh` is the monotone-enforced form from
    `statsmodels.multipletests`, which is what the published panel used; `p_adj_bh_plain` is the
    unenforced `p * m / rank` form used for the Fig. 5b family.

    Args:
        table: output of transition_table for the same grouping
        spec: one entry of GROUPINGS

    Returns:
        DataFrame with one row per (transition, comparison)
    """
    lookup = {(r["transition"], r["group"]): r for _, r in table.iterrows()}
    rows = []
    for _, _, name in TRANSITIONS:
        for comparison, i_a, i_b in COMPARISONS:
            a = lookup[(name, spec["labels"][i_a])]
            b = lookup[(name, spec["labels"][i_b])]
            n_a, o_a = a["n_at_start"], a["n_reaching"]
            n_b, o_b = b["n_at_start"], b["n_reaching"]
            _, p_raw = proportions_ztest([o_a, o_b], [n_a, n_b])
            rr = (o_a / n_a) / (o_b / n_b)
            se = np.sqrt((n_a - o_a) / (o_a * n_a) + (n_b - o_b) / (o_b * n_b))
            rows.append(
                {
                    "grouping": spec["name"],
                    "transition": name,
                    "comparison": COMPARISON_LABELS[comparison],
                    "comparison_key": comparison,
                    "group_a": a["group"],
                    "group_b": b["group"],
                    "rate_a": a["rate"],
                    "rate_b": b["rate"],
                    "risk_ratio": rr,
                    "rr_ci_low": float(np.exp(np.log(rr) - 1.96 * se)),
                    "rr_ci_high": float(np.exp(np.log(rr) + 1.96 * se)),
                    "p_raw": float(p_raw),
                }
            )
    out = pd.DataFrame(rows)
    out["p_adj_bh"] = multipletests(out["p_raw"], method="fdr_bh")[1]
    p = out["p_raw"].to_numpy()
    order = np.argsort(p, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(p) + 1)
    out["p_adj_bh_plain"] = np.minimum(p * len(p) / rank, 1.0)
    return out


rates, tests = {}, {}
for name, spec in GROUPINGS.items():
    spec["name"] = name
    rates[name] = transition_table(UNIVERSE, spec)
    tests[name] = pairwise_table(rates[name], spec)
    print(f"{name}: group sizes " + ", ".join(
        f"{r['group']}={r['group_n']:,}" for _, r in rates[name].drop_duplicates("group").iterrows()
    ))
''',
)

add(
    "markdown",
    """\
### Step 1 control — the published panel must reproduce first

Hard assertions against every number printed in the manuscript for the therapeutic-area panel. If
any of these fail the universe or the method differs from what we think and nothing below is valid.
""",
)

add(
    "code",
    '''\
ta_rates = rates["TAs"].set_index(["group", "transition"])
ta_tests = tests["TAs"].set_index(["transition", "comparison_key"])

# group sizes
assert list(rates["TAs"].drop_duplicates("group")["group_n"]) == [6578, 9705, 2197]

# printed transition rates
for group, transition, expected in [
    ("1 TA", "Phase I\\u2192II", 82.9),
    ("2\\u20135 TAs", "Phase I\\u2192II", 84.2),
    ("\\u22656 TAs", "Phase I\\u2192II", 82.6),
    ("1 TA", "Phase II\\u2192III", 57.0),
    ("2\\u20135 TAs", "Phase II\\u2192III", 53.5),
    ("\\u22656 TAs", "Phase II\\u2192III", 49.8),
    ("1 TA", "Phase III\\u2192approval", 29.4),
    ("2\\u20135 TAs", "Phase III\\u2192approval", 31.2),
]:
    got = round(ta_rates.loc[(group, transition), "rate"] * 100, 1)
    assert got == expected, (group, transition, got, expected)

# printed Phase II->III risk ratios, CIs and adjusted P values
for comparison, rr, lo, hi in [
    ("med_vs_low", 0.94, 0.91, 0.97),
    ("high_vs_low", 0.87, 0.83, 0.92),
    ("high_vs_med", 0.93, 0.88, 0.98),
]:
    row = ta_tests.loc[("Phase II\\u2192III", comparison)]
    assert (round(row["risk_ratio"], 2), round(row["rr_ci_low"], 2), round(row["rr_ci_high"], 2)) == (rr, lo, hi), (
        comparison,
        row[["risk_ratio", "rr_ci_low", "rr_ci_high"]].to_dict(),
    )

p_ii_iii = ta_tests.loc["Phase II\\u2192III", "p_adj_bh"]
assert f"{p_ii_iii['med_vs_low']:.0e}" == "3e-04", p_ii_iii["med_vs_low"]
assert p_ii_iii["high_vs_low"] < 0.001, p_ii_iii["high_vs_low"]
assert round(p_ii_iii["high_vs_med"], 3) == 0.010, p_ii_iii["high_vs_med"]

print("STEP 1 CONTROL PASSED \\u2014 published therapeutic-area panel reproduced to printed precision")
print(f"  Medium vs Low  P_adj = {p_ii_iii['med_vs_low']:.3e}")
print(f"  High vs Low    P_adj = {p_ii_iii['high_vs_low']:.3e}")
print(f"  High vs Medium P_adj = {p_ii_iii['high_vs_med']:.3e}")
''',
)

add(
    "markdown",
    """\
### BH convention

Fig. 5b's printed FDRs use plain BH (`p * m / rank`, no monotone enforcement). The published
Extended Data Fig. 8 values instead come from `statsmodels.multipletests(method="fdr_bh")`, which
does enforce monotonicity. For this family the two conventions agree on every printed value, so the
control passes either way; the monotone form is kept because it is what the published notebook ran.
Both columns are carried into the CSVs.
""",
)

add(
    "code",
    '''\
for name in GROUPINGS:
    t = tests[name]
    disagree = t[~np.isclose(t["p_adj_bh"], t["p_adj_bh_plain"])]
    print(f"{name}: {len(disagree)} of {len(t)} tests differ between BH conventions")
    if len(disagree):
        print(
            disagree[["transition", "comparison", "p_raw", "p_adj_bh", "p_adj_bh_plain"]].to_string(index=False)
        )
    flips = t[(t["p_adj_bh"] < 0.05) != (t["p_adj_bh_plain"] < 0.05)]
    print(f"  significance calls that change at 0.05: {len(flips)}")
''',
)

add(
    "markdown",
    """\
### Results — both panels
""",
)

add(
    "code",
    '''\
for name in GROUPINGS:
    print(f"=== {name} " + "=" * 70)
    wide = rates[name].pivot(index="transition", columns="group", values="rate")
    wide = wide[GROUPINGS[name]["labels"]].reindex([t[2] for t in TRANSITIONS])
    print((wide * 100).round(1).to_string())
    show = tests[name][
        ["transition", "comparison", "risk_ratio", "rr_ci_low", "rr_ci_high", "p_raw", "p_adj_bh"]
    ].copy()
    show["risk_ratio"] = show.apply(
        lambda r: f"{r['risk_ratio']:.2f} ({r['rr_ci_low']:.2f}\\u2013{r['rr_ci_high']:.2f})", axis=1
    )
    show["signif"] = np.where(show["p_adj_bh"] < 0.05, "*", "")
    print(
        show.drop(columns=["rr_ci_low", "rr_ci_high"])
        .assign(p_raw=lambda x: x["p_raw"].map("{:.3e}".format), p_adj_bh=lambda x: x["p_adj_bh"].map("{:.3e}".format))
        .to_string(index=False)
    )
    print()
''',
)

add(
    "markdown",
    """\
### Two-panel figure

Both panels share colours, fonts, bar geometry and y-axis range. Group *n* is printed above each
bar; significance brackets are drawn only where the BH-adjusted P < 0.05. A single figure-level
legend names the tier and both bin definitions, since the bins differ between panels.
""",
)

add(
    "code",
    '''\
TIER_COLOURS = ["#c6dbef", "#6baed6", "#08519c"]
BAR_WIDTH = 0.8 / 3
N_OFFSET = 0.025
BRACKET_STEP = 0.06
BRACKET_BASE = 0.075


def p_to_stars(p):
    """Star notation for an adjusted P value, or None when not significant."""
    for threshold, stars in [(0.0001, "****"), (0.001, "***"), (0.01, "**"), (0.05, "*")]:
        if p < threshold:
            return stars
    return None


def bracket_plan(test_table):
    """Bracket geometry for one panel: adjacent pairs on the low level, spanning pair above.

    Args:
        test_table: pairwise_table output for one grouping

    Returns:
        list of (transition_index, x_slot_a, x_slot_b, level, stars)
    """
    plan = []
    for t_i, (_, _, name) in enumerate(TRANSITIONS):
        sub = test_table[test_table["transition"] == name].set_index("comparison_key")
        adjacent = [("med_vs_low", 0, 1), ("high_vs_med", 1, 2)]
        spanning = [("high_vs_low", 0, 2)]
        level = 0
        for key, a, b in adjacent:
            stars = p_to_stars(sub.loc[key, "p_adj_bh"])
            if stars:
                plan.append((t_i, a, b, 0, stars))
                level = 1
        for key, a, b in spanning:
            stars = p_to_stars(sub.loc[key, "p_adj_bh"])
            if stars:
                plan.append((t_i, a, b, level, stars))
    return plan


def slot_x(t_i, group_i):
    """Bar centre for transition t_i, group group_i."""
    return t_i - 0.4 + BAR_WIDTH * (group_i + 0.5)


# shared y-limit: the tallest bracket across both panels plus headroom
y_top = 0.97
plans = {name: bracket_plan(tests[name]) for name in GROUPINGS}
for name in GROUPINGS:
    tab = rates[name]
    for t_i, _, _, level, _ in plans[name]:
        base = tab[tab["transition"] == TRANSITIONS[t_i][2]]["ci_high"].max()
        y_top = max(y_top, base + BRACKET_BASE + BRACKET_STEP * level + 0.05)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)

for ax, (name, spec), tag in zip(axes, GROUPINGS.items(), ["a", "b"], strict=True):
    tab = rates[name].set_index(["group", "transition"])
    for g_i, group in enumerate(spec["labels"]):
        for t_i, (_, _, transition) in enumerate(TRANSITIONS):
            row = tab.loc[(group, transition)]
            x = slot_x(t_i, g_i)
            ax.bar(
                x,
                row["rate"],
                width=BAR_WIDTH,
                color=TIER_COLOURS[g_i],
                edgecolor="none",
                label=group if t_i == 0 else None,
            )
            ax.errorbar(
                x=x,
                y=row["rate"],
                yerr=[[row["rate"] - row["ci_low"]], [row["ci_high"] - row["rate"]]],
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=3,
                zorder=10,
            )
            ax.text(
                x,
                row["ci_high"] + N_OFFSET,
                f"n={int(row['n_at_start']):,}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    for t_i, a, b, level, stars in plans[name]:
        base = rates[name].query("transition == @TRANSITIONS[@t_i][2]")["ci_high"].max()
        y = base + BRACKET_BASE + BRACKET_STEP * level
        x1, x2 = slot_x(t_i, a), slot_x(t_i, b)
        ax.plot([x1, x1, x2, x2], [y - 0.012, y, y, y - 0.012], lw=1.0, color="black", clip_on=False)
        ax.text((x1 + x2) / 2, y + 0.003, stars, ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(range(len(TRANSITIONS)))
    ax.set_xticklabels([t[2] for t in TRANSITIONS])
    ax.set_xlim(-0.5, len(TRANSITIONS) - 0.5)
    ax.set_xlabel("Transition")
    ax.set_title(f"grouped by {'therapeutic areas' if name == 'TAs' else 'gPS (unique diseases)'}", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.06, 1.04, tag, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")

axes[0].set_ylabel("Transition probability")
axes[0].set_ylim(0.2, y_top)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=TIER_COLOURS[i], edgecolor="none")
    for i in range(3)
]
legend_labels = [
    f"{tier} \\u2014 {GROUPINGS['TAs']['labels'][i]} (a) / {GROUPINGS['gPS']['labels'][i]} (b)"
    for i, tier in enumerate(["Low", "Medium", "High"])
]
fig.legend(
    legend_handles,
    legend_labels,
    title="Pleiotropy",
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=9,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle("Target-disease pair transition success probability by pleiotropy", y=1.0)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("extended_figure_8-r1.pdf", dpi=300, bbox_inches="tight")
plt.show()
''',
)

add(
    "markdown",
    """\
### Sensitivity — gPS ≤ 5 / 6–9 / ≥ 10

Recorded for the record only, not shown in the figure. The figure keeps gPS 1 / 2–9 / ≥ 10 because
the ≥ 10 high cut point is what the referee asked for and what Fig. 5b uses, and because a Low bin of
gPS 1 keeps the Low tier comparable to the 1-TA Low tier in panel (a).
""",
)

add(
    "code",
    '''\
sens_spec = {
    "name": "gPS_sensitivity",
    "column": "uniqueDiseases",
    "edges": [0, 5, 9, float("inf")],
    "labels": ["gPS \\u22645", "gPS 6\\u20139", "gPS \\u226510"],
}
sens_rates = transition_table(UNIVERSE, sens_spec)
sens_tests = pairwise_table(sens_rates, sens_spec)

print("group sizes " + ", ".join(
    f"{r['group']}={r['group_n']:,}" for _, r in sens_rates.drop_duplicates("group").iterrows()
))
sens_wide = sens_rates.pivot(index="transition", columns="group", values="rate")
print((sens_wide[sens_spec["labels"]].reindex([t[2] for t in TRANSITIONS]) * 100).round(1).to_string())
print(
    sens_tests[["transition", "comparison", "risk_ratio", "rr_ci_low", "rr_ci_high", "p_raw", "p_adj_bh"]]
    .assign(
        p_raw=lambda x: x["p_raw"].map("{:.3e}".format),
        p_adj_bh=lambda x: x["p_adj_bh"].map("{:.3e}".format),
    )
    .round(3)
    .to_string(index=False)
)
''',
)

add(
    "markdown",
    """\
### Write the `-r1` result tables
""",
)

add(
    "code",
    '''\
rates_out = pd.concat([rates["TAs"], rates["gPS"], sens_rates], ignore_index=True)
tests_out = pd.concat([tests["TAs"], tests["gPS"], sens_tests], ignore_index=True)

rates_path = path_to_intermediate_data_folder + "ed8_phase_transition_rates-r1.csv"
tests_path = path_to_intermediate_data_folder + "ed8_phase_transition_tests-r1.csv"
rates_out.to_csv(rates_path, index=False)
tests_out.to_csv(tests_path, index=False)
print(f"wrote {rates_path} ({len(rates_out)} rows)")
print(f"wrote {tests_path} ({len(tests_out)} rows)")
print("wrote extended_figure_8-r1.pdf")

thin = rates_out[rates_out["n_at_start"] < 1500][
    ["grouping", "group", "transition", "n_at_start", "n_reaching"]
]
print("\\nthin cells (fewer than 1,500 pairs at the start phase):")
print(thin.to_string(index=False))
''',
)


def main() -> None:
    """Patch the notebook: redirect the control PDF, drop empties, replace the tagged section."""
    nb = json.loads(NB.read_text())

    nb["cells"][0]["source"] = HEADER_MD.splitlines(keepends=True)

    control = "".join(nb["cells"][13]["source"])
    assert 'fig.savefig("extended_figure_8.pdf"' in control, "control cell already redirected?"
    nb["cells"][13]["source"] = control.replace(
        'fig.savefig("extended_figure_8.pdf"',
        'fig.savefig("extended_figure_8_control_ta_only-r1.pdf"',
    ).splitlines(keepends=True)

    kept = [
        c
        for c in nb["cells"]
        if TAG not in c.get("metadata", {}).get("tags", [])
        and "".join(c["source"]).strip() != ""
    ]

    for cell_type, source in CELLS:
        cell = {
            "cell_type": cell_type,
            "metadata": {"tags": [TAG]},
            "source": source.splitlines(keepends=True),
        }
        if cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        kept.append(cell)

    nb["cells"] = kept
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"patched {NB} -> {len(kept)} cells ({len(CELLS)} appended)")


if __name__ == "__main__":
    main()
