"""
Plot eval/episode_success_any from a W&B project.

W&B run configs are empty for this project, so env/agent/seed are parsed
from the run metadata args stored in the local wandb/ directory.

Usage:
    # single env
    python ece567/plot_success.py --env ant --output ant.png
    # all 4 envs in a 1x4 grid
    python ece567/plot_success.py --output all_envs.png
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
AGENTS = {"crl", "ppo", "sac", "td3"}
ENVS = ["reacher", "pusher_hard", "humanoid", "ant"]
ENV_LABELS = {"ant": "Ant", "humanoid": "Humanoid", "pusher_hard": "Pusher Hard", "reacher": "Reacher"}

# metric key -> (wandb key, normalise_by, y-axis label)
METRICS = {
    "any":  ("eval/episode_success_any",  None, "Success Rate"),
    "easy": ("eval/episode_success_easy", 1001, "Success Rate (easy, 2m threshold)"),
}

DEFAULT_WANDB_DIR = os.path.join(os.path.dirname(__file__), "..", "wandb")


def build_id_map(wandb_dir: str) -> dict[str, tuple[str, str, int]]:
    """Return {run_id: (env, agent, seed)} parsed from local wandb-metadata.json files."""
    id_map = {}
    for meta_path in glob.glob(os.path.join(wandb_dir, "run-*/files/wandb-metadata.json")):
        run_id = os.path.basename(os.path.dirname(os.path.dirname(meta_path))).split("-")[-1]
        with open(meta_path) as f:
            meta = json.load(f)
        args = meta.get("args", [])
        env, agent, seed = None, None, 0
        i = 0
        while i < len(args):
            if args[i] == "--env" and i + 1 < len(args):
                env = args[i + 1]; i += 2
            elif args[i] == "--seed" and i + 1 < len(args):
                seed = int(args[i + 1]); i += 2
            elif args[i] in AGENTS:
                agent = args[i]; i += 1
            else:
                i += 1
        if env and agent:
            id_map[run_id] = (env, agent, seed)
    return id_map


def fetch_all_runs(project: str, wandb_dir: str, metric_key: str, normalise_by):
    """Return {env: {agent: {seed: [(step, value), ...]}}} skipping still-running runs.

    When multiple finished runs share the same (env, agent, seed), only the most
    recently created one is used.
    """
    id_map = build_id_map(wandb_dir)
    api = wandb.Api()

    # First pass: pick the newest finished run per (env, agent, seed).
    best: dict[tuple, tuple] = {}  # key -> (created_at, run)
    for run in api.runs(project):
        if run.state == "running":
            print(f"  Skipping still-running run {run.id}")
            continue
        info = id_map.get(run.id)
        if info is None:
            continue
        key = info  # (env, agent, seed)
        if key not in best or run.created_at > best[key][0]:
            best[key] = (run.created_at, run)

    # Second pass: fetch history only from the selected runs.
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (env, agent, seed), (_, run) in best.items():
        for row in run.history(keys=[metric_key, STEP_KEY], samples=500, pandas=False):
            step = row.get(STEP_KEY)
            val = row.get(metric_key)
            if step is not None and val is not None:
                if normalise_by:
                    val = val / normalise_by
                data[env][agent][seed].append((int(step), float(val)))
    return data


def align_seeds(seed_data: dict[int, list[tuple[int, float]]]):
    """Interpolate seeds onto a shared step grid (clipped to their common range).

    Returns (steps, mat) where mat is (n_seeds, n_steps).
    """
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
    """Interquartile mean over seed axis (axis 0)."""
    q25 = np.percentile(mat, 25, axis=0)
    q75 = np.percentile(mat, 75, axis=0)
    mask = (mat >= q25) & (mat <= q75)
    counts = np.where(mask.sum(axis=0) == 0, 1, mask.sum(axis=0))
    return (mat * mask).sum(axis=0) / counts


def drop_worst_seed(seed_data: dict) -> dict:
    """Remove the seed with the lowest mean value over the final 20% of its steps."""
    if len(seed_data) <= 1:
        return seed_data
    scores = {}
    for seed, pts in seed_data.items():
        if not pts:
            continue
        sorted_pts = sorted(pts)
        cutoff = max(1, int(len(sorted_pts) * 0.8))
        scores[seed] = np.mean([v for _, v in sorted_pts[cutoff:]])
    worst = min(scores, key=scores.get)
    print(f"    dropping worst seed {worst} (score={scores[worst]:.4f})")
    return {s: pts for s, pts in seed_data.items() if s != worst}


def draw_env(ax, env: str, agent_data: dict, cmap):
    """Draw one environment's subplot onto ax."""
    for i, agent in enumerate(sorted(agent_data)):
        seed_data = agent_data[agent]
        if env == "pusher_hard":
            seed_data = drop_worst_seed(seed_data)
        steps, mat = align_seeds(seed_data)
        if steps.size == 0:
            continue
        n_seeds = mat.shape[0]
        center = iqm(mat)
        se = mat.std(axis=0) / np.sqrt(n_seeds)
        color = cmap(i)
        ax.plot(steps, center, label=agent.upper(), color=color, linewidth=2)
        ax.fill_between(steps, center - se, center + se, color=color, alpha=0.2)
        print(f"  {env}/{agent}: {n_seeds} seed(s), steps {int(steps[0])}-{int(steps[-1])}")

    ax.set_title(ENV_LABELS.get(env, env), fontsize=16)
    ax.set_xlabel("Environment Steps", fontsize=15)
    ax.tick_params(labelsize=15)
    ax.grid(True, alpha=0.3)


def plot_all(project: str, wandb_dir: str, metric: str, output: str | None):
    metric_key, normalise_by, ylabel = METRICS[metric]
    print(f"Fetching all runs from project '{project}' (metric={metric_key}) ...")
    all_data = fetch_all_runs(project, wandb_dir, metric_key, normalise_by)

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, env in zip(axes, ENVS):
        agent_data = all_data.get(env, {})
        if not agent_data:
            print(f"  No data for env='{env}'")
            ax.set_title(ENV_LABELS.get(env, env))
            continue
        draw_env(ax, env, agent_data, cmap)

    axes[0].set_ylabel(ylabel, fontsize=15)
    axes[0].legend(loc="upper left", fontsize=13)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def plot_single(project: str, env: str, wandb_dir: str, metric: str, output: str | None):
    metric_key, normalise_by, ylabel = METRICS[metric]
    print(f"Fetching runs for env='{env}' (metric={metric_key}) ...")
    all_data = fetch_all_runs(project, wandb_dir, metric_key, normalise_by)
    agent_data = all_data.get(env, {})

    if not agent_data:
        print(f"No runs found for env='{env}'.")
        return

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_env(ax, env, agent_data, cmap)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", fontsize=11)
    ax.autoscale(axis="y")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="jaxgcrl")
    parser.add_argument("--env", default=None, help="Single env to plot; omit for all 4 envs")
    parser.add_argument("--wandb_dir", default=DEFAULT_WANDB_DIR)
    parser.add_argument("--metric", default="any", choices=list(METRICS),
                        help="'any' = strict 0.5m threshold (default); 'easy' = 2m threshold")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    wandb_dir = os.path.abspath(args.wandb_dir)
    if args.env:
        plot_single(args.project, args.env, wandb_dir, args.metric, args.output)
    else:
        plot_all(args.project, wandb_dir, args.metric, args.output)


if __name__ == "__main__":
    main()
