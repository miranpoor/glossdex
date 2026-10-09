"""Report tables for the core benchmark, with paired bootstrap CIs (from results/test-*.json).

    python summarize.py        # -> results/summary-test.md

For each collection: every system's shown-set metrics, then glossdex (text profile, set once on
dev) minus each baseline on the same queries: task score (answerable F2 + probe rejection),
answerable-only F2, and nDCG@10. 10,000 paired resamples, seeded per comparison.
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
BOOT = 10000
REF = "glossdex"
NAMES = {"scifact": "SciFact (test)", "nfcorpus": "NFCorpus (test)", "fiqa": "FiQA (test)"}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def paired(a, b, rng):
    d = [x - y for x, y in zip(a, b)]
    m = mean(d)
    bs = sorted(mean([rng.choice(d) for _ in d]) for _ in range(BOOT))
    return m, bs[int(0.025 * BOOT)], bs[int(0.975 * BOOT) - 1]


def f(x, pct=True):
    return f"{100 * x:.1f}" if pct else f"{x:.2f}"


def figure():
    """Figure 1: task score per collection, glossdex vs the baselines. Validated categorical
    palette in fixed order; no per-bar error bars (comparisons are paired; CIs are in the tables)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False})
    data = {}
    for n in ("scifact", "nfcorpus", "fiqa"):
        d = json.load(open(os.path.join(RES, f"test-{n}.json"), encoding="utf-8"))
        data[n] = d
    keys = list(data["scifact"]["rows"])

    series = [(REF, "glossdex (set once on other collections)", "#2a78d6"),
              (next(k for k in keys if k.endswith("+ threshold")), "Dense, top 5 + threshold (set on other collections)", "#1f3a5f"),
              ("dense · threshold", "Dense, threshold (set on other collections)", "#7f8c8d"),
              (next(k for k in keys if k.startswith("adaptive-k") and k.endswith("(set on dev)")),
               "Adaptive-k (buffer set on other collections)", "#9b59b6"),
              ("dense · top5", "Dense, top 5", "#eb6834"),
              ("hybrid · top5", "Hybrid (RRF), top 5", "#eda100")]
    labels = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa": "FiQA"}

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    groups = list(data)
    w = 0.8 / len(series)
    for i, (s, lab, col) in enumerate(series):
        xs = [g - 0.4 + w * (i + 0.5) for g in range(len(groups))]
        ys = [100 * mean([data[n]["rows"][s][q[0]]["task"] for q in data[n]["queries"]]) for n in groups]
        ax.bar(xs, ys, w * 0.92, color=col, label=lab, linewidth=0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([labels[n] for n in groups])
    ax.set_ylabel("Task score (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.30), fontsize=7.5)
    out = os.path.join(RES, "figures")
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig1_task_score.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(out, "fig1_task_score.png"))


def main():
    L = ["# Core benchmark: glossdex on public text collections (test splits)\n",
         "Same EmbeddingGemma-300M vectors for every system. glossdex: result-set settings chosen once on three "
         "other collections (SciDocs, ArguAna, CQADupStack-webmasters; text_profile.json), then frozen. The "
         "threshold and Adaptive-k baselines' settings were chosen the same way (baseline_settings.json). No test "
         "collection was used for any setting. Probes: 50 queries from an unrelated collection per "
         "test collection, which the corpus cannot answer. BEIR judgments are sparse, so precision is a lower bound "
         "for every system alike.\n"]
    for n in ("scifact", "nfcorpus", "fiqa"):
        p = os.path.join(RES, f"test-{n}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        rows, qs = d["rows"], d["queries"]
        ans = [q[0] for q in qs if not q[3]]
        prb = [q[0] for q in qs if q[3]]
        L.append(f"\n## {NAMES[n]}: {len(ans)} answerable queries, {len(prb)} probes\n")
        L.append("| system · cutoff | task | F2 | recall | precision | wrong/q | shown/q | probes: nothing shown | answerable: nothing shown | nDCG@10 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        order = sorted(rows, key=lambda k: -mean([rows[k][q]["task"] for q in ans + prb]))
        # A system is set in bold only when it leads BOTH the task score and F2 (the recall-precision
        # combination on answerable queries) in this collection.
        task_of = {k: round(mean([rows[k][q]["task"] for q in ans + prb]), 6) for k in rows}
        f2_of = {k: round(mean([rows[k][q]["F2"] for q in ans]), 6) for k in rows}
        leader = {k for k in rows if task_of[k] == max(task_of.values()) and f2_of[k] == max(f2_of.values())}
        for k in order:
            r = rows[k]
            name = f"**{k}**" if k in leader else k
            L.append(f"| {name} | {f(mean([r[q]['task'] for q in ans + prb]))} | {f(mean([r[q]['F2'] for q in ans]))} | "
                     f"{f(mean([r[q]['recall'] for q in ans]))} | {f(mean([r[q]['precision'] for q in ans]))} | "
                     f"{f(mean([r[q]['wrong'] for q in ans]), False)} | {f(mean([r[q]['shown'] for q in ans]), False)} | "
                     f"{f(mean([r[q]['rejected'] for q in prb]))} | {f(mean([1.0 if r[q]['shown'] == 0 else 0.0 for q in ans]))} | "
                     f"{f(mean([r[q]['nDCG@10'] for q in ans]))} |")
        if REF not in rows:
            continue
        L.append(f"\nPaired differences, `{REF}` minus each (points, 95% CI):\n")
        L.append("| vs | task | F2 (answerable only) | nDCG@10 |")
        L.append("|---|---|---|---|")
        ref = rows[REF]
        for k in order:
            if k.startswith("glossdex"):
                continue
            o = rows[k]
            cells = []
            for metric, ids in (("task", ans + prb), ("F2", ans), ("nDCG@10", ans)):
                # one fixed seed per comparison, so an interval never depends on which other rows exist
                m, lo, hi = paired([ref[q][metric] for q in ids], [o[q][metric] for q in ids],
                                   random.Random(f"{n}|{k}|{metric}"))
                cells.append(f"{100 * m:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f})")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
    figure()
    out = os.path.join(RES, "summary-test.md")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
