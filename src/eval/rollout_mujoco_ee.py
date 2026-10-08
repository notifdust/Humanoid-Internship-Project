"""Integrate predicted EE deltas in a minimal MuJoCo scene and write an MP4.

This is the Windows-safe sim path while full LIBERO (robosuite) may need WSL.
Actions: [dx, dy, dz, gripper, ...] from retarget / SmolVLA space.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np


SCENE_XML = """
<mujoco model="ee_proxy">
  <option timestep="0.033" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.85 0.85 0.85" rgb2="0.7 0.7 0.7"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance="0.1"/>
    <material name="mug" rgba="0.9 0.1 0.55 1"/>
    <material name="plate" rgba="0.95 0.95 0.95 1"/>
    <material name="ee" rgba="0.2 0.45 0.85 1"/>
    <material name="grip_open" rgba="0.2 0.8 0.3 1"/>
    <material name="grip_closed" rgba="0.9 0.3 0.2 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="table" type="plane" size="0.6 0.6 0.02" material="grid"/>
    <body name="plate" pos="0.12 -0.05 0.01">
      <geom type="cylinder" size="0.08 0.008" material="plate"/>
    </body>
    <body name="mug" pos="-0.1 0.08 0.04">
      <freejoint/>
      <geom type="cylinder" size="0.035 0.04" material="mug" mass="0.2"/>
    </body>
    <body name="ee" pos="0 0 0.25">
      <joint name="ee_x" type="slide" axis="1 0 0" limited="true" range="-0.45 0.45"/>
      <joint name="ee_y" type="slide" axis="0 1 0" limited="true" range="-0.45 0.45"/>
      <joint name="ee_z" type="slide" axis="0 0 1" limited="true" range="0.02 0.45"/>
      <geom name="ee_body" type="sphere" size="0.025" material="ee" mass="0.05"/>
      <geom name="ee_tip" type="capsule" fromto="0 0 0 0 0 -0.04" size="0.01" material="grip_open"/>
      <camera name="ego" pos="0.15 -0.25 0.35" xyaxes="1 0.2 0 0 0.6 1" fovy="50"/>
    </body>
  </worldbody>
</mujoco>
"""


def load_actions(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        ep = np.load(path, allow_pickle=True)
        return np.asarray(ep["action"], dtype=np.float64)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    raise SystemExit(f"Unsupported actions file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actions",
        type=Path,
        default=Path("data/processed/episodes/pick_place_mug_41.npz"),
        help="NPZ episode (uses GT actions) or .npy of predicted actions",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/eval/mujoco_ee_41.mp4"))
    parser.add_argument("--scale", type=float, default=2.5, help="scale delta actions into meters")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    import mujoco

    actions = load_actions(args.actions)
    if actions.ndim != 2 or actions.shape[1] < 3:
        raise SystemExit(f"Expected [T, >=3] actions, got {actions.shape}")

    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(SCENE_XML)
        xml_path = f.name

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    # Start above mug side of table
    data.qpos[model.joint("ee_x").qposadr[0]] = -0.05
    data.qpos[model.joint("ee_y").qposadr[0]] = 0.05
    data.qpos[model.joint("ee_z").qposadr[0]] = 0.22
    mujoco.mj_forward(model, data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (args.width, args.height)
    )

    tip_id = model.geom("ee_tip").id
    trail = []
    for i, act in enumerate(actions):
        jx = model.joint("ee_x").qposadr[0]
        jy = model.joint("ee_y").qposadr[0]
        jz = model.joint("ee_z").qposadr[0]
        data.qpos[jx] = np.clip(data.qpos[jx] + float(act[0]) * args.scale, -0.45, 0.45)
        data.qpos[jy] = np.clip(data.qpos[jy] + float(act[1]) * args.scale, -0.45, 0.45)
        data.qpos[jz] = np.clip(data.qpos[jz] + float(act[2]) * args.scale, 0.02, 0.45)

        grip = float(act[3]) if act.shape[0] > 3 else 0.0
        # Tint tip by gripper (closed = red-ish)
        if grip > 0.5:
            model.geom_rgba[tip_id] = np.array([0.9, 0.3, 0.2, 1.0], dtype=np.float32)
        else:
            model.geom_rgba[tip_id] = np.array([0.2, 0.8, 0.3, 1.0], dtype=np.float32)

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="ego")
        rgb = renderer.render()
        trail.append([float(data.qpos[jx]), float(data.qpos[jy]), float(data.qpos[jz]), grip])

        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame,
            f"t={i} grip={'CLOSE' if grip > 0.5 else 'open'}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
        )
        writer.write(frame)

    writer.release()
    meta = {
        "actions": str(args.actions),
        "frames": len(actions),
        "scale": args.scale,
        "out": str(args.out),
        "ee_end": trail[-1] if trail else None,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} frames={len(actions)}")


if __name__ == "__main__":
    main()
