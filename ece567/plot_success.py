"""
Plot eval/episode_success_any from a W&B project.
2×4 grid: top row = reacher, pusher_hard, humanoid, ant;
          bottom row = ant_u_maze, ant_big_maze, ant_ball (soccer), ant_push.
SAC and TD3 are split by use_her flag into separate curves.
SCCRLV2, SCCRLV4, SCCRLV6 are excluded. Only finished runs are plotted.

Usage:
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

ALL_AGENT_NAMES = {"crl", "ppo", "sac", "td3", "sccrlv2", "sccrlv3", "sccrlv4", "sccrlv5", "sccrlv6"}
AGENTS_IGNORE = {"sccrlv2", "sccrlv4", "sccrlv6"}

ENVS_TOP = ["reacher", "pusher_hard", "humanoid", "ant"]
ENVS_BOT = ["ant_u_maze", "ant_big_maze", "ant_ball", "ant_push"]
ENVS = ENVS_TOP + ENVS_BOT

ENV_LABELS = {
    "ant": "Ant",
    "humanoid": "Humanoid",
    "pusher_hard": "Pusher Hard",
    "reacher": "Reacher",
    "ant_u_maze": "Ant U-Maze",
    "ant_big_maze": "Ant Big Maze",
    "ant_ball": "Ant Soccer",
    "ant_push": "Ant Push",
}

AGENT_LABELS = {
    "crl": "CRL",
    "ppo": "PPO",
    "sac": "SAC",
    "sac_her": "SAC+HER",
    "td3": "TD3",
    "td3_her": "TD3+HER",
    "sccrlv3": "SCCRLv3",
    "sccrlv5": "SCCRL (ours)",
}

# Fixed global agent ordering determines color assignment (tab10 index).
AGENT_ORDER = ["crl", "ppo", "sac", "sac_her", "sccrlv3", "sccrlv5", "td3", "td3_her"]

# Color overrides for agents whose default tab10 color isn't ideal.
AGENT_COLOR_OVERRIDES = {
    "sccrlv5": "#17becf",  # teal; replaces the default brown
}

# Seeds to exclude per (env, agent) pair.
SEEDS_IGNORE: dict[tuple, set] = {
    ("ant_u_maze", "crl"): {2, 3},
}

# metric key -> (wandb key, normalise_by, y-axis label)
METRICS = {
    "any":  ("eval/episode_success_any",  None, "Success Rate"),
    "easy": ("eval/episode_success_easy", 1001, "Success Rate (easy, 2m threshold)"),
}

DEFAULT_WANDB_DIR = os.path.join(os.path.dirname(__file__), "..", "wandb")

# For these (env, agent, seed) keys, use the LEAST recently created finished run.
LEAST_RECENT_KEYS = {("ant_big_maze", "crl", 0), ("ant_u_maze", "crl", 0)}


def build_id_map(wandb_dir: str) -> dict[str, tuple[str, str, int]]:
    """Return {run_id: (env, agent, seed)} parsed from local wandb-metadata.json files.

    SAC/TD3 with --use_her become 'sac_her'/'td3_her'.
    Runs matching AGENTS_IGNORE are excluded.
    """
    id_map = {}
    for meta_path in glob.glob(os.path.join(wandb_dir, "run-*/files/wandb-metadata.json")):
        run_id = os.path.basename(os.path.dirname(os.path.dirname(meta_path))).split("-")[-1]
        with open(meta_path) as f:
            meta = json.load(f)
        args = meta.get("args", [])
        env, agent, seed, use_her = None, None, 0, False
        i = 0
        while i < len(args):
            if args[i] == "--env" and i + 1 < len(args):
                env = args[i + 1]; i += 2
            elif args[i] == "--seed" and i + 1 < len(args):
                seed = int(args[i + 1]); i += 2
            elif args[i] == "--use_her":
                next_is_value = (i + 1 < len(args) and
                                 not args[i + 1].startswith("--") and
                                 args[i + 1].lower() in ("true", "false", "1", "0", "yes", "no"))
                if next_is_value:
                    use_her = args[i + 1].lower() in ("true", "1", "yes"); i += 2
                else:
                    use_her = True; i += 1
            elif args[i].lower() in ALL_AGENT_NAMES:
                agent_name = args[i].lower()
                if agent_name not in AGENTS_IGNORE:
                    agent = agent_name
                i += 1
            else:
                i += 1
        if env and agent:
            if use_her and agent in ("sac", "td3"):
                agent = f"{agent}_her"
            id_map[run_id] = (env, agent, seed)
    return id_map


def fetch_all_runs(project: str, wandb_dir: str, metric_key: str, normalise_by):
    """Return {env: {agent: {seed: [(step, value), ...]}}} using only finished runs.

    Most recent finished run per (env, agent, seed), except LEAST_RECENT_KEYS
    which use the least recently created finished run.
    """
    id_map = build_id_map(wandb_dir)
    api = wandb.Api()

    best: dict[tuple, tuple] = {}  # key -> (created_at, run)
    for run in api.runs(project):
        if run.state != "finished":
            print(f"  Skipping {run.state} run {run.id}")
            continue
        info = id_map.get(run.id)
        if info is None:
            continue
        env, agent, seed = info
        if seed in SEEDS_IGNORE.get((env, agent), set()):
            continue
        key = info  # (env, agent, seed)
        if key in LEAST_RECENT_KEYS:
            if key not in best or run.created_at < best[key][0]:
                best[key] = (run.created_at, run)
        else:
            if key not in best or run.created_at > best[key][0]:
                best[key] = (run.created_at, run)

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (env, agent, seed), (_, run) in best.items():
        for row in run.history(keys=[metric_key, STEP_KEY], samples=500, pandas=False):
            step = row.get(STEP_KEY)
            val = row.get(metric_key)
            if step is not None and val is not None and step <= 50_000_000:
                if normalise_by:
                    val = val / normalise_by
                data[env][agent][seed].append((int(step), float(val)))
    return data


def align_seeds(seed_data: dict[int, list[tuple[int, float]]]):
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


def drop_worst_seed(seed_data: dict) -> dict:
    if len(seed_data) <= 1:
        return seed_data
    scores = {}
    for seed, pts in seed_data.items():
        if not pts:
            continue
        sorted_pts = sorted(pts)
        cutoff = max(1, int(len(sorted_pts) * 0.8))
        scores[seed] = np.mean([v for _, v in sorted_pts[cutoff:]])
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    worst = ranked[0][0]
    print(f"    dropping worst seed {worst} (score={scores[worst]:.4f})")
    return {s: pts for s, pts in seed_data.items() if s != worst}


def draw_env(ax, env: str, agent_data: dict, agent_colors: dict):
    for agent in sorted(agent_data):
        seed_data = agent_data[agent]
        if env == "pusher_hard":
            seed_data = drop_worst_seed(seed_data)
        steps, mat = align_seeds(seed_data)
        if steps.size == 0:
            continue
        n_seeds = mat.shape[0]
        center = iqm(mat) if n_seeds > 2 else mat.mean(axis=0)
        se = mat.std(axis=0) / np.sqrt(n_seeds)
        color = agent_colors[agent]
        label = AGENT_LABELS.get(agent, agent.upper())
        steps_m = steps / 1e6
        ax.plot(steps_m, center, label=label, color=color, linewidth=2)
        ax.fill_between(steps_m, np.clip(center - se, 0, 1), np.clip(center + se, 0, 1),
                        color=color, alpha=0.2)
        print(f"  {env}/{agent}: {n_seeds} seed(s), steps {int(steps[0])}-{int(steps[-1])}")

    ax.set_title(ENV_LABELS.get(env, env), fontsize=26)
    ax.set_xlabel("Environment Steps (×10⁶)", fontsize=22)
    ax.tick_params(labelsize=20)

    x_max = ax.get_xlim()[1]
    ax.set_xticks(np.arange(0, x_max + 1, 10))

    ax.autoscale(axis="y")
    yticks = [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    non_neg_ticks = [t for t in yticks if t >= 0]
    if non_neg_ticks:
        tick_spacing = non_neg_ticks[1] - non_neg_ticks[0] if len(non_neg_ticks) > 1 else non_neg_ticks[0]
        bottom = non_neg_ticks[0] - 0.15 * tick_spacing
        ax.set_ylim(bottom=bottom)
        ax.set_yticks(non_neg_ticks)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)


def plot_all(project: str, wandb_dir: str, metric: str, output: str | None):
    metric_key, normalise_by, ylabel = METRICS[metric]
    print(f"Fetching all runs from project '{project}' (metric={metric_key}) ...")
    all_data = fetch_all_runs(project, wandb_dir, metric_key, normalise_by)

    cmap = plt.get_cmap("tab10")
    # Fixed color per agent so every subplot uses the same palette.
    agent_colors = {agent: cmap(i) for i, agent in enumerate(AGENT_ORDER)}
    # Fallback color for any agent not in AGENT_ORDER.
    for i, agent in enumerate(sorted(set(a for env_data in all_data.values() for a in env_data) - set(AGENT_ORDER))):
        agent_colors[agent] = cmap(len(AGENT_ORDER) + i)
    agent_colors.update(AGENT_COLOR_OVERRIDES)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    agents_present: set[str] = set()
    for row, envs_row in enumerate([ENVS_TOP, ENVS_BOT]):
        for col, env in enumerate(envs_row):
            ax = axes[row, col]
            agent_data = all_data.get(env, {})
            if not agent_data:
                print(f"  No data for env='{env}'")
                ax.set_title(ENV_LABELS.get(env, env), fontsize=26)
                continue
            agents_present.update(agent_data.keys())
            draw_env(ax, env, agent_data, agent_colors)

    for row in range(2):
        axes[row, 0].set_ylabel(ylabel, fontsize=22)

    # Legend in top-left subplot with handles for all agents present in any env.
    ordered_agents = [a for a in AGENT_ORDER if a in agents_present]
    ordered_agents += sorted(agents_present - set(AGENT_ORDER))
    if "sccrlv5" in ordered_agents:
        ordered_agents = ["sccrlv5"] + [a for a in ordered_agents if a != "sccrlv5"]
    legend_handles = [
        plt.Line2D([0], [0], color=agent_colors[a], linewidth=2,
                   label=AGENT_LABELS.get(a, a.upper()))
        for a in ordered_agents
    ]
    axes[0, 0].legend(handles=legend_handles, loc="lower right",
                      bbox_to_anchor=(1.0, 0.2), fontsize=17)

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
    parser.add_argument("--metric", default="any", choices=list(METRICS),
                        help="'any' = strict 0.5m threshold (default); 'easy' = 2m threshold")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    plot_all(args.project, os.path.abspath(args.wandb_dir), args.metric, args.output)


if __name__ == "__main__":
    main()
