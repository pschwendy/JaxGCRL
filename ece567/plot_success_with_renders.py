"""
2x4 grid: top row = env render frames, bottom row = success rate plots.
Renders are produced by resetting each Brax env (EGL headless).
Plots are the same IQM ± SE curves as plot_success.py.

Usage:
    python ece567/plot_success_with_renders.py --output all_envs.png
"""

import os
os.environ["MUJOCO_GL"] = "egl"  # must be set before any mujoco/brax import

import argparse
import glob
import json
from collections import defaultdict

import jax
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import wandb

from jaxgcrl.utils.env import create_env

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


# fraction of default free-camera distance to zoom in per env
ZOOM = {"ant": 0.10, "humanoid": 0.10, "pusher_hard": 0.5, "reacher": 0.4}
# camera elevation in degrees (default is -45; less negative = more side-on)
CAM_ELEVATION = {"ant": -15, "humanoid": -25, "pusher_hard": -45, "reacher": -45}


# ── rendering ────────────────────────────────────────────────────────────────

def render_env_frame(env_name: str, width: int = 480, height: int = 480) -> np.ndarray:
    """Reset the Brax env and return a zoomed RGB frame via MuJoCo EGL renderer."""
    env = create_env(env_name)
    state = env.reset(jax.random.PRNGKey(0))
    ps = state.pipeline_state

    renderer = mujoco.Renderer(env.sys.mj_model, height=height, width=width)
    d = mujoco.MjData(env.sys.mj_model)
    d.qpos, d.qvel = ps.q, ps.qd
    mujoco.mj_forward(env.sys.mj_model, d)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.sys.mj_model, cam)
    cam.distance *= ZOOM.get(env_name, 0.5)
    cam.elevation = CAM_ELEVATION.get(env_name, -45)
    if env_name == "humanoid":
        cam.lookat[2] += 0.8  # shift focus point upward so full body is centred

    renderer.update_scene(d, camera=cam)
    frame = renderer.render()
    renderer.close()
    return frame


# ── W&B data fetching ─────────────────────────────────────────────────────────

def build_id_map(wandb_dir: str) -> dict[str, tuple[str, str, int]]:
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


# ── statistics ────────────────────────────────────────────────────────────────

def align_seeds(seed_data):
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


def iqm(mat):
    q25, q75 = np.percentile(mat, 25, axis=0), np.percentile(mat, 75, axis=0)
    mask = (mat >= q25) & (mat <= q75)
    counts = np.where(mask.sum(axis=0) == 0, 1, mask.sum(axis=0))
    return (mat * mask).sum(axis=0) / counts


# ── plotting ──────────────────────────────────────────────────────────────────

def draw_plot(ax, env: str, agent_data: dict, cmap, show_legend: bool):
    for i, agent in enumerate(sorted(agent_data)):
        steps, mat = align_seeds(agent_data[agent])
        if steps.size == 0:
            continue
        n_seeds = mat.shape[0]
        center = iqm(mat)
        se = mat.std(axis=0) / np.sqrt(n_seeds)
        color = cmap(i)
        steps_m = steps / 1e6
        ax.plot(steps_m, center, label=agent.upper(), color=color, linewidth=2)
        ax.fill_between(steps_m, np.clip(center - se, 0, 1), np.clip(center + se, 0, 1),
                        color=color, alpha=0.2)
        print(f"  {env}/{agent}: {n_seeds} seed(s)")

    ax.set_xlabel("Environment Steps (×10⁶)", fontsize=22)
    ax.tick_params(labelsize=20)

    # x ticks every 10M steps
    x_max = ax.get_xlim()[1]
    ax.set_xticks(np.arange(0, x_max + 1, 10))

    # y: let matplotlib autoscale, then nudge bottom slightly below the lowest tick
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
    if show_legend:
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.2), fontsize=17)


def plot_all(project: str, wandb_dir: str, metric: str, output: str | None):
    metric_key, normalise_by, ylabel = METRICS[metric]
    print("Rendering env frames ...")
    frames = {env: render_env_frame(env) for env in ENVS}

    print(f"Fetching W&B runs from '{project}' (metric={metric_key}) ...")
    all_data = fetch_all_runs(project, wandb_dir, metric_key, normalise_by)

    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(24, 9))
    plot_axes = []

    for col, env in enumerate(ENVS):
        label = ENV_LABELS[env]

        # ── top row: render ──
        ax_img = fig.add_subplot(2, 4, col + 1)
        frame = frames[env]
        h, w = frame.shape[:2]
        # Crop to 3:2 by trimming top/bottom (keep centre)
        target_h = int(w * 2 / 3)
        margin = (h - target_h) // 2
        ax_img.imshow(frame[margin:margin + target_h, :], aspect="auto")
        ax_img.axis("off")
        ax_img.set_title(label, fontsize=26)

        # ── bottom row: plot ──
        ax_plot = fig.add_subplot(2, 4, col + 5)
        plot_axes.append(ax_plot)
        agent_data = all_data.get(env, {})
        if agent_data:
            draw_plot(ax_plot, env, agent_data, cmap, show_legend=(col == 0))
        else:
            print(f"  No data for env='{env}'")

    plot_axes[0].set_ylabel(ylabel, fontsize=22)

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
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    plot_all(args.project, os.path.abspath(args.wandb_dir), args.metric, args.output)


if __name__ == "__main__":
    main()
