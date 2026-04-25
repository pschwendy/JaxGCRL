"""
Ablation study: 1×3 grid over ant, ant_u_maze, ant_big_maze.

Ablation chain (each step removes one component from SCCRL):
  sccrlv5                              → full SCCRL
  sccrlv6                              → no improvement filter
  sccrlv6 --cvae_alignment_coeff 0.0  → no improvement filter, no alignment term
  sccrlv7                              → random interpolated subgoals

Usage:
    python ece567/plot_ablation.py --output ablation.png
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import wandb

STEP_KEY = "_step"
MAX_STEPS = 50_000_000

ABLATION_AGENTS = {"sccrlv5", "sccrlv6", "sccrlv7"}

ENVS = ["ant", "ant_u_maze", "ant_big_maze"]
ENV_LABELS = {
    "ant":         "Ant",
    "ant_u_maze":  "Ant U-Maze",
    "ant_big_maze": "Ant Big Maze",
}

# Internal key → display label (ordered for legend)
AGENT_ORDER = ["sccrlv5", "sccrlv6", "sccrlv6_noalign", "sccrlv7"]
AGENT_LABELS = {
    "sccrlv5":       "SCCRL (full)",
    "sccrlv6":       "− improvement filter",
    "sccrlv6_noalign": "− filter & alignment",
    "sccrlv7":       "interpolated subgoals",
}

cmap = plt.get_cmap("tab10")
AGENT_COLORS = {agent: cmap(i) for i, agent in enumerate(AGENT_ORDER)}

METRICS = {
    "any":  ("eval/episode_success_any",  None, "Success Rate"),
    "easy": ("eval/episode_success_easy", 1001, "Success Rate (easy, 2m threshold)"),
}

DEFAULT_WANDB_DIR = os.path.join(os.path.dirname(__file__), "..", "wandb")


def build_id_map(wandb_dir: str) -> dict[str, tuple[str, str, int]]:
    """Return {run_id: (env, agent_key, seed)}.

    sccrlv6 with --cvae_alignment_coeff 0.0 is keyed as 'sccrlv6_noalign'.
    """
    id_map = {}
    for meta_path in glob.glob(os.path.join(wandb_dir, "run-*/files/wandb-metadata.json")):
        run_id = os.path.basename(os.path.dirname(os.path.dirname(meta_path))).split("-")[-1]
        with open(meta_path) as f:
            meta = json.load(f)
        args = meta.get("args", [])
        env, agent, seed = None, None, 0
        cvae_alignment_coeff = None
        i = 0
        while i < len(args):
            if args[i] == "--env" and i + 1 < len(args):
                env = args[i + 1]; i += 2
            elif args[i] == "--seed" and i + 1 < len(args):
                seed = int(args[i + 1]); i += 2
            elif args[i] == "--cvae_alignment_coeff" and i + 1 < len(args):
                cvae_alignment_coeff = float(args[i + 1]); i += 2
            elif args[i].lower() in ABLATION_AGENTS:
                agent = args[i].lower(); i += 1
            else:
                i += 1
        if env and agent and env in ENVS:
            if agent == "sccrlv6" and cvae_alignment_coeff is not None and cvae_alignment_coeff == 0.0:
                agent = "sccrlv6_noalign"
            id_map[run_id] = (env, agent, seed)
    return id_map


def fetch_all_runs(project: str, wandb_dir: str, metric_key: str, normalise_by):
    """Return {env: {agent: {seed: [(step, value), ...]}}} for finished runs only."""
    id_map = build_id_map(wandb_dir)
    api = wandb.Api()

    best: dict[tuple, tuple] = {}
    for run in api.runs(project):
        if run.state != "finished":
            continue
        info = id_map.get(run.id)
        if info is None:
            continue
        key = info
        if key not in best or run.created_at > best[key][0]:
            best[key] = (run.created_at, run)

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (env, agent, seed), (_, run) in best.items():
        if seed != 0:
            continue
        for row in run.history(keys=[metric_key, STEP_KEY], samples=500, pandas=False):
            step = row.get(STEP_KEY)
            val = row.get(metric_key)
            if step is not None and val is not None and step <= MAX_STEPS:
                if normalise_by:
                    val = val / normalise_by
                data[env][agent][seed].append((int(step), float(val)))
    return data


def align_seeds(seed_data: dict):
    sorted_seeds = {s: sorted(pts) for s, pts in seed_data.items() if pts}
    if not sorted_seeds:
        return np.array([]), np.array([])
    ref_seed = max(sorted_seeds, key=lambda s: len(sorted_seeds[s]))
    ref_xs = np.array([x for x, _ in sorted_seeds[ref_seed]], dtype=float)
    x_min = max(pts[0][0] for pts in sorted_seeds.values())
    x_max = min(pts[-1][0] for pts in sorted_seeds.values())
    steps = ref_xs[(ref_xs >= x_min) & (ref_xs <= x_max)]
    if steps.size == 0:
        return np.array([]), np.array([])
    rows = []
    for pts in sorted_seeds.values():
        xs, ys = zip(*pts)
        rows.append(np.interp(steps, np.array(xs, dtype=float), np.array(ys, dtype=float)))
    return steps, np.array(rows)


def iqm(mat: np.ndarray) -> np.ndarray:
    q25 = np.percentile(mat, 25, axis=0)
    q75 = np.percentile(mat, 75, axis=0)
    mask = (mat >= q25) & (mat <= q75)
    counts = np.where(mask.sum(axis=0) == 0, 1, mask.sum(axis=0))
    return (mat * mask).sum(axis=0) / counts


def draw_env(ax, env: str, agent_data: dict, show_legend: bool):
    for agent in AGENT_ORDER:
        seed_data = agent_data.get(agent)
        if not seed_data:
            continue
        steps, mat = align_seeds(seed_data)
        if steps.size == 0:
            continue
        color = AGENT_COLORS[agent]
        steps_m = steps / 1e6
        center = mat[0]
        ax.plot(steps_m, center, label=AGENT_LABELS[agent], color=color, linewidth=2)
        print(f"  {env}/{agent}: seed 0")

    ax.set_title(ENV_LABELS[env], fontsize=26)
    ax.set_xlabel("Environment Steps (×10⁶)", fontsize=22)
    ax.tick_params(labelsize=20)

    x_max = ax.get_xlim()[1]
    ax.set_xticks(np.arange(0, x_max + 1, 10))

    ax.autoscale(axis="y")
    yticks = [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    non_neg_ticks = [t for t in yticks if t >= 0]
    if non_neg_ticks:
        tick_spacing = non_neg_ticks[1] - non_neg_ticks[0] if len(non_neg_ticks) > 1 else non_neg_ticks[0]
        ax.set_ylim(bottom=non_neg_ticks[0] - 0.15 * tick_spacing)
        ax.set_yticks(non_neg_ticks)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=17)


def plot_all(project: str, wandb_dir: str, metric: str, output: str | None):
    metric_key, normalise_by, ylabel = METRICS[metric]
    print(f"Fetching ablation runs from '{project}' (metric={metric_key}) ...")
    all_data = fetch_all_runs(project, wandb_dir, metric_key, normalise_by)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for col, env in enumerate(ENVS):
        agent_data = all_data.get(env, {})
        if not agent_data:
            print(f"  No data for env='{env}'")
            axes[col].set_title(ENV_LABELS[env], fontsize=26)
            continue
        draw_env(axes[col], env, agent_data, show_legend=(col == 0))

    axes[0].set_ylabel(ylabel, fontsize=22)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="jaxgcrl")
    parser.add_argument("--wandb_dir", default=DEFAULT_WANDB_DIR)
    parser.add_argument("--metric", default="any", choices=list(METRICS))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    plot_all(args.project, os.path.abspath(args.wandb_dir), args.metric, args.output)


if __name__ == "__main__":
    main()
