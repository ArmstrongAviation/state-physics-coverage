"""Generate paired Cube trajectories under controlled mechanism interventions.

Each group reuses one source initial state and one action sequence. Only the
simulator mechanism changes between variants, so held-out compositions can test
whether a world model learned dynamics instead of trajectory appearance.
"""

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401: register HDF5 compression filters
import mujoco
import numpy as np
from ogbench.manipspace.envs.cube_env import CubeEnv


VARIANTS = (
    "nominal", "mass_low", "mass_high", "friction_low", "friction_high", "delay_2",
    "mass_high_delay_2", "friction_low_delay_2",
)


def _object_body_id(env):
    return env.model.body("object_0").id


def _object_geom_ids(env):
    return [env.model.geom(name).id for name in ("object_0",)]


def _apply_variant(env, variant, mass_low_scale, mass_high_scale, friction_low_scale, friction_high_scale):
    body_id = _object_body_id(env)
    if variant in ("mass_low",):
        env.model.body_mass[body_id] *= mass_low_scale
        env.model.body_inertia[body_id] *= mass_low_scale
    elif variant in ("mass_high", "mass_high_delay_2"):
        env.model.body_mass[body_id] *= mass_high_scale
        env.model.body_inertia[body_id] *= mass_high_scale
    elif variant in ("friction_low", "friction_low_delay_2"):
        for gid in _object_geom_ids(env):
            env.model.geom_friction[gid, 0] = friction_low_scale
    elif variant == "friction_high":
        for gid in _object_geom_ids(env):
            env.model.geom_friction[gid, 0] = friction_high_scale
    elif variant not in ("nominal", "delay_2", "mass_high_delay_2", "friction_low_delay_2"):
        raise ValueError(f"unknown variant: {variant}")
    if variant in ("mass_low", "mass_high", "mass_high_delay_2"):
        mujoco.mj_setConst(env.model, env.data)


def _write_variant(out_path, variant, groups, actions, qpos, qvel, pixels, block_pos, contact, ee_pos,
                   source_path, frame_skip):
    with h5py.File(out_path, "w") as f:
        f.attrs["source_h5"] = str(source_path)
        f.attrs["variant"] = variant
        f.attrs["frame_skip"] = frame_skip
        f.create_dataset("group_id", data=groups[:, 0], dtype="i8")
        f.create_dataset("source_episode", data=groups[:, 1], dtype="i8")
        f.create_dataset("source_start", data=groups[:, 2], dtype="i8")
        f.create_dataset("initial_qpos", data=qpos, compression="gzip", compression_opts=1)
        f.create_dataset("initial_qvel", data=qvel, compression="gzip", compression_opts=1)
        f.create_dataset("action", data=actions, compression="gzip", compression_opts=1)
        shape = pixels.shape
        f.create_dataset("pixels", data=pixels, chunks=(1, 1, shape[-3], shape[-2], shape[-1]),
                         compression="gzip", compression_opts=1, shuffle=True)
        f.create_dataset("block_pos", data=block_pos, compression="gzip", compression_opts=1)
        f.create_dataset("gripper_contact", data=contact, compression="gzip", compression_opts=1)
        f.create_dataset("effector_pos", data=ee_pos, compression="gzip", compression_opts=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/data/chenyizi/le-wm/datasets/lewm-cube/cube_single_expert.h5")
    ap.add_argument("--out-dir", default="/data/chenyizi/lewm-data-study/counterfactual")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="optional Cube train_specs manifest for contact/phase-focused starts")
    ap.add_argument("--preserve-manifest-order", action="store_true",
                    help="use the first manifest rows in order instead of resampling them")
    ap.add_argument("--mass-low-scale", type=float, default=0.5)
    ap.add_argument("--mass-high-scale", type=float, default=2.0)
    ap.add_argument("--friction-low-scale", type=float, default=0.25)
    ap.add_argument("--friction-high-scale", type=float, default=1.75)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    if args.horizon % args.frame_skip:
        raise ValueError("horizon must be divisible by frame-skip")
    bad = set(args.variants) - set(VARIANTS)
    if bad:
        raise ValueError(f"unsupported variants: {sorted(bad)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.source, "r") as src:
        ep_len = src["ep_len"][:]
        ep_offset = src["ep_offset"][:]
        valid = np.where(ep_len >= args.horizon + 1)[0]
        if args.manifest:
            specs = np.load(args.manifest, allow_pickle=False)["train_specs"].astype(np.int64)
            specs = np.asarray([(ep, start) for ep, start in specs
                                if ep in set(valid) and start + args.horizon + 1 <= ep_len[ep]], dtype=np.int64)
            if len(specs) == 0:
                raise ValueError("manifest has no starts long enough for requested horizon")
            if args.preserve_manifest_order:
                if args.episodes > len(specs):
                    raise ValueError("preserve-manifest-order requires enough manifest rows")
                chosen_idx = np.arange(args.episodes, dtype=np.int64)
            else:
                chosen_idx = rng.choice(len(specs), size=args.episodes, replace=len(specs) < args.episodes)
            chosen_specs = specs[chosen_idx]
            groups = [(gid, int(ep), int(start)) for gid, (ep, start) in enumerate(chosen_specs)]
        else:
            chosen = rng.choice(valid, size=min(args.episodes, len(valid)), replace=False)
            groups = []
            for gid, ep in enumerate(chosen):
                max_start = int(ep_len[ep]) - args.horizon - 1
                start = int(rng.integers(0, max_start + 1))
                groups.append((gid, int(ep), start))
        groups = np.asarray(groups, dtype=np.int64)
        actions = np.stack([src["action"][int(ep_offset[e]) + s:int(ep_offset[e]) + s + args.horizon]
                            for _, e, s in groups]).astype(np.float32)
        qpos = np.stack([src["qpos"][int(ep_offset[e]) + s] for _, e, s in groups]).astype(np.float64)
        qvel = np.stack([src["qvel"][int(ep_offset[e]) + s] for _, e, s in groups]).astype(np.float64)

    n_lat = args.horizon // args.frame_skip + 1
    for variant in args.variants:
        print(f"generating {variant}: groups={len(groups)} horizon={args.horizon}", flush=True)
        env = CubeEnv("single", ob_type="pixels", mode="data_collection", width=224, height=224)
        env.reset(seed=args.seed)
        _apply_variant(env, variant, args.mass_low_scale, args.mass_high_scale,
                       args.friction_low_scale, args.friction_high_scale)
        pixels = np.empty((len(groups), n_lat, 224, 224, 3), dtype=np.uint8)
        block_pos = np.empty((len(groups), n_lat, 3), dtype=np.float32)
        contact = np.empty((len(groups), n_lat, 1), dtype=np.float32)
        ee_pos = np.empty((len(groups), n_lat, 3), dtype=np.float32)
        for i in range(len(groups)):
            env.set_state(qpos[i], qvel[i])
            env._prev_qpos = env.data.qpos.copy()
            env._prev_qvel = env.data.qvel.copy()
            delayed = [np.zeros(5, dtype=np.float32), np.zeros(5, dtype=np.float32)]
            for k in range(args.horizon + 1):
                if k % args.frame_skip == 0:
                    j = k // args.frame_skip
                    pixels[i, j] = env.render()
                    block_pos[i, j] = env.data.joint("object_joint_0").qpos[:3]
                    contact[i, j] = env.compute_ob_info()["proprio/gripper_contact"]
                    ee_pos[i, j] = env.data.site_xpos[env.model.site("ur5e/robotiq/pinch").id]
                if k == args.horizon:
                    break
                command = actions[i, k]
                if variant in ("delay_2", "mass_high_delay_2", "friction_low_delay_2"):
                    delayed.append(command)
                    command = delayed.pop(0)
                env.step(command)
            if (i + 1) % max(1, len(groups) // 8) == 0:
                print(f"  {i + 1}/{len(groups)}", flush=True)
        _write_variant(out_dir / f"cube_{variant}.h5", variant, groups, actions, qpos, qvel,
                       pixels, block_pos, contact, ee_pos, args.source, args.frame_skip)
        env.close()
        print(f"wrote {out_dir / f'cube_{variant}.h5'}", flush=True)


if __name__ == "__main__":
    main()
