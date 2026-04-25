"""
2×3 grid of environment renders, no titles or labels.
Top row:    reacher, pusher_hard, humanoid
Bottom row: ant, ant_big_maze, ant_push

Usage:
    python ece567/plot_renders.py --output renders.png
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse

import jax
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from jaxgcrl.utils.env import create_env

ENVS = [
    ["reacher", "pusher_hard", "humanoid"],
    ["ant",     "ant_big_maze", "ant_push"],
]

ZOOM = {
    "ant":         0.10,
    "humanoid":    0.10,
    "pusher_hard": 0.5,
    "reacher":     0.4,
    "ant_big_maze": 0.5,
    "ant_push":    0.5,
}
CAM_ELEVATION = {
    "ant":         -15,
    "humanoid":    -25,
    "pusher_hard": -45,
    "reacher":     -45,
    "ant_big_maze": -90,
    "ant_push":    -90,
}
# Override lookat target (world-space XYZ) to centre the view on the scene.
CAM_LOOKAT = {
    "ant_big_maze": [14.0, 14.0, 0.0],  # center of 8×8 grid at 4.0 scaling
}


def render_env_frame(env_name: str, width: int = 480, height: int = 480) -> np.ndarray:
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
    if env_name in CAM_LOOKAT:
        cam.lookat[:] = CAM_LOOKAT[env_name]
    cam.elevation = CAM_ELEVATION.get(env_name, -45)
    if env_name == "humanoid":
        cam.lookat[2] += 0.8

    renderer.update_scene(d, camera=cam)
    frame = renderer.render()
    renderer.close()
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for row, env_row in enumerate(ENVS):
        for col, env_name in enumerate(env_row):
            print(f"Rendering {env_name} ...")
            frame = render_env_frame(env_name)
            axes[row, col].imshow(frame, aspect="auto")
            axes[row, col].axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
