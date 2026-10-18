"""Collect every stage-3 result into tables and figures for the report.

    python scripts/analyze.py                                  # tables + figures from results/
    python scripts/analyze.py --corpus-cache data/cache/v100000_val7_test7   # + distribution plots

Reads results/stage3_rl/<task>/<config>/metrics.json (and topk_test.npz for
the distribution figure); writes results/summary.csv, results/summary.md and
figures/*.png. Pure post-processing: no GPU, no ClearML.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical order (light surface); text never wears series colour.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 7,
})


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.glob("stage3_rl/*/*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        cfg = m["config"]
        runs.append({
            "task": path.parts[-3], "name": path.parts[-2], "dir": path.parent,
            "is_base": cfg["beta"] == "base", "reward": cfg["reward"], "beta": cfg["beta"],
            "uncertainty": cfg.get("uncertainty", 0), "horizon": cfg.get("horizon", 1),
            "seed": cfg.get("seed", 0), "steps": cfg.get("steps", 0), "m": m,
        })
    return runs


def flat(run: dict) -> dict:
    t = run["m"]["test"]
    u, b, s = t["user_level"], t["user_level"]["behaviour@10"], t["slate_reward@10"]
    r = run["reward"]
    return {
        "task": run["task"], "config": run["name"], "reward": r, "beta": run["beta"],
        "uncertainty": run["uncertainty"], "horizon": run["horizon"], "seed": run["seed"],
        "steps": run["steps"],
        "recall@10": u["recall@10"], "ndcg@10": u["ndcg@10"], "map@10": u["map@10"],
        "recall@100": u["recall@100"], "next_hit@10": t["next_item"]["all"]["hit@10"],
        "proxy_paper": s["proxy_paper"], "gold_paper": s["gold_paper"],
        "proxy_target": s[f"proxy_{r}"], "gold_target": s[f"gold_{r}"],
        "ensemble_std": s["proxy_disagreement"],
        "popularity": b["avg_popularity"], "track_len_s": b["avg_track_length_s"],
        "coverage": b["coverage"], "top10_share": b["top10_share"],
        "jaccard": b["pairwise_jaccard"], "repeat_share": b["repeat_share"],
        "artists_per_list": b["unique_artists_per_list"], "entropy_bits": b["entropy_bits"],
    }


def base_for(runs: list[dict], reward: str) -> dict | None:
    for r in runs:
        if r["is_base"] and r["reward"] == reward:
            return r
    return None


def write_tables(rows: list[dict], out: Path) -> None:
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = ["config", "recall@10", "ndcg@10", "map@10", "recall@100", "proxy_target",
            "gold_target", "track_len_s", "popularity", "coverage", "jaccard", "repeat_share"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    seen = set()
    for r in rows:
        key = (r["config"], r["task"]) if not str(r["config"]).startswith("base_") else r["config"]
        if key in seen:
            continue
        seen.add(key)
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = f"{v:.0f}" if c == "popularity" else (f"{v:.1f}" if c == "track_len_s" else f"{v:.4f}")
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fig_pareto(runs, fig_dir: Path) -> None:
    """Relevance vs. reward over the KL weight, for the paper reward."""
    base = base_for(runs, "paper")
    sweep = sorted([r for r in runs if r["reward"] == "paper" and not r["is_base"]
                    and r["horizon"] == 1 and not r["uncertainty"] and r["seed"] == 0],
                   key=lambda r: -r["beta"])
    if not base or not sweep:
        return
    b = flat(base)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key, title in ((axes[0], "proxy_paper", "Proxy RM (optimised)"),
                           (axes[1], "gold_paper", "Gold RM (held-out users)")):
        xs = [100 * (flat(r)[key] / b[key] - 1) for r in sweep]
        ys = [flat(r)["recall@10"] for r in sweep]
        ax.plot(xs, ys, color=SERIES[0], marker="o", zorder=3)
        ax.scatter([0], [b["recall@10"]], color=INK_2, marker="s", zorder=4, s=50)
        ax.annotate("base", (0, b["recall@10"]), textcoords="offset points", xytext=(-8, 8),
                    color=INK_2, fontsize=9)
        # Points that sit on top of each other share one label.
        groups: list[list] = []
        for x, y, r in zip(xs, ys, sweep):
            if groups and abs(groups[-1][0][0] - x) < 0.35 and abs(groups[-1][0][1] - y) < 0.002:
                groups[-1].append((x, y, r))
            else:
                groups.append([(x, y, r)])
        for i, g in enumerate(groups):
            betas = ", ".join(f"{r['beta']:g}" for _, _, r in g)
            x, y, _ = g[-1]
            # Alternate above/below so neighbouring labels never collide.
            ax.annotate(f"β={betas}", (x, y), textcoords="offset points",
                        xytext=(6, -16) if i % 2 == 0 else (6, 8), color=INK_2, fontsize=9)
        # Second seed of the same settings, as open markers.
        reps = [r for r in runs if r["reward"] == "paper" and not r["is_base"] and r["seed"] != 0
                and r["horizon"] == 1 and not r["uncertainty"]]
        if reps:
            ax.scatter([100 * (flat(r)[key] / b[key] - 1) for r in reps],
                       [flat(r)["recall@10"] for r in reps], facecolors="none",
                       edgecolors=SERIES[1], linewidths=1.5, s=60, zorder=5, label="seed 1")
            ax.legend(loc="lower left", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Δ expected reward of top-10, % vs base")
    axes[0].set_ylabel("Recall@10 (test week)")
    fig.suptitle("Relevance vs. reward across the KL weight β (reward = like + 0.1·ratio)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "pareto_beta.png", dpi=150)
    plt.close(fig)


def fig_reward_variants(runs, fig_dir: Path) -> None:
    variants = [r for r in runs if not r["is_base"] and r["horizon"] == 1 and not r["uncertainty"]
                and r["seed"] == 0 and r["beta"] == 0.05]
    variants = sorted(variants, key=lambda r: r["reward"])
    if not variants:
        return
    labels, d_reward, d_recall, d_len, d_rep = [], [], [], [], []
    for r in variants:
        b = base_for(runs, r["reward"])
        if not b:
            continue
        f, fb = flat(r), flat(b)
        labels.append(r["reward"])
        d_reward.append(100 * (f["gold_target"] / fb["gold_target"] - 1))
        d_recall.append(100 * (f["recall@10"] / fb["recall@10"] - 1))
        d_len.append(f["track_len_s"] - fb["track_len_s"])
        d_rep.append(100 * (f["repeat_share"] - fb["repeat_share"]))
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    panels = ((d_reward, "Δ own reward (gold RM), %"), (d_recall, "Δ Recall@10, %"),
              (d_len, "Δ mean track length, s"), (d_rep, "Δ share already heard, pp"))
    for ax, (vals, title) in zip(axes, panels):
        ax.barh(y, vals, color=SERIES[0], height=0.55)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_title(title, fontsize=10)
        for yi, v in zip(y, vals):
            ax.annotate(f"{v:+.1f}", (v, yi), textcoords="offset points",
                        xytext=(4 if v >= 0 else -4, -3), ha="left" if v >= 0 else "right",
                        color=INK_2, fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.3)
    axes[0].set_yticks(y, labels)
    fig.suptitle("Same KL weight (β = 0.05), different rewards: change vs. the base model",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "reward_variants.png", dpi=150)
    plt.close(fig)


def fig_multistep(runs, fig_dir: Path) -> None:
    base = base_for(runs, "paper")
    collapsed = next((r for r in runs if r["reward"] == "paper" and r["beta"] == 0
                      and r["horizon"] == 1 and not r["uncertainty"] and r["seed"] == 0), None)
    if not base:
        return
    series = [("base, predicted feedback", base, "predicted", SERIES[0]),
              ("base, hidden feedback", base, "unknown", SERIES[1])]
    if collapsed:
        series.append(("RL β=0, predicted feedback", collapsed, "predicted", SERIES[2]))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    metrics = (("gold", "Gold-RM reward of generated item"),
               ("precision", "Item is in user's real positives"),
               ("disagreement", "Proxy ensemble disagreement (std)"))
    steps = np.arange(1, 6)
    for ax, (key, title) in zip(axes, metrics):
        for label, run, fb, color in series:
            vals = [s[key] for s in run["m"]["multistep_test"][fb]]
            ax.plot(steps, vals, color=color, marker="o", label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("generation step")
        ax.set_xticks(steps)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Greedy 5-step generation from the true state at the start of the test week",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(fig_dir / "multistep.png", dpi=150)
    plt.close(fig)


def fig_training(runs, fig_dir: Path) -> None:
    wanted = (0.0, 0.001, 0.005, 0.05)
    chosen = sorted([r for r in runs if r["reward"] == "paper" and not r["is_base"]
                     and r["horizon"] == 1 and not r["uncertainty"] and r["seed"] == 0
                     and r["beta"] in wanted and r["m"].get("history")], key=lambda r: r["beta"])
    if not chosen:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for i, r in enumerate(chosen):
        h = r["m"]["history"]
        steps = [x["step"] for x in h]
        for ax, key in zip(axes, ("reward", "kl", "entropy")):
            ax.plot(steps, [max(x[key], 1e-5) if key == "kl" else x[key] for x in h],
                    color=SERIES[i], label=f"β={r['beta']:g}")
    axes[0].set_title("Proxy reward of sampled items", fontsize=10)
    axes[1].set_title("KL(policy ‖ base), nats", fontsize=10)
    axes[1].set_yscale("log")
    axes[2].set_title("Policy entropy, nats", fontsize=10)
    for ax in axes:
        ax.set_xlabel("RL step")
    axes[0].legend(fontsize=8)
    fig.suptitle("RL training dynamics (reward = like + 0.1·ratio)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "training.png", dpi=150)
    plt.close(fig)


def fig_distributions(runs, fig_dir: Path, corpus_cache: Path, root: Path) -> None:
    """ECDF of track length and popularity of recommended items (needs the corpus cache)."""
    arrays = np.load(corpus_cache / "corpus.npz")
    meta = json.loads((corpus_cache / "corpus_meta.json").read_text(encoding="utf-8"))
    item, ts = arrays["item"], arrays["ts"]
    length = arrays["item_length"]
    pop = np.bincount(item[ts < meta["val_start"]], minlength=len(length)).astype(float)

    picks = [("base", base_for(runs, "paper"))]
    for label, reward, beta in (("RL β=0 (paper)", "paper", 0), ("RL seconds, β=0.05", "seconds", 0.05),
                                ("RL ratio, β=0.05", "ratio", 0.05)):
        picks.append((label, next((r for r in runs if r["reward"] == reward and r["beta"] == beta
                                   and r["horizon"] == 1 and not r["uncertainty"]
                                   and r["seed"] == 0 and not r["is_base"]), None)))
    base_topk = next(iter(sorted(root.glob("stage1_base/*/topk_test.npz"))), Path("missing"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, (label, run) in enumerate(picks):
        path = (run["dir"] / "topk_test.npz") if run and not run["is_base"] else base_topk
        if not path.exists():
            continue
        topk = np.load(path)["topk"][:, :10].ravel()
        for ax, values in ((axes[0], length[topk]), (axes[1], np.log10(pop[topk] + 1))):
            v = np.sort(values)
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=SERIES[i], label=label)
    axes[0].set_title("Track length of top-10 items", fontsize=10)
    axes[0].set_xlabel("seconds")
    axes[0].set_xlim(0, 600)
    axes[1].set_title("Popularity of top-10 items", fontsize=10)
    axes[1].set_xlabel("log10(train-period listens + 1)")
    axes[0].set_ylabel("cumulative share")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("What the recommended lists are made of", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "distributions.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Tables and figures from results/.")
    p.add_argument("--results", default="results")
    p.add_argument("--figures", default="figures")
    p.add_argument("--corpus-cache", default=None,
                   help="Folder with corpus.npz (enables the distribution figure).")
    args = p.parse_args()
    root, fig_dir = Path(args.results), Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"No stage-3 results under {root}.")
    rows = [flat(r) for r in runs]
    write_tables(rows, root)
    fig_pareto(runs, fig_dir)
    fig_reward_variants(runs, fig_dir)
    fig_multistep(runs, fig_dir)
    fig_training(runs, fig_dir)
    if args.corpus_cache:
        fig_distributions(runs, fig_dir, Path(args.corpus_cache), root)
    print(f"{len(runs)} runs -> {root / 'summary.md'}, figures in {fig_dir}/")


if __name__ == "__main__":
    main()
