"""Audit pairing and mechanism separation in generated Cube counterfactuals."""

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="counterfactual_v1")
    ap.add_argument("--out", default="reports/counterfactual_audit.json")
    args = ap.parse_args()
    root = Path(args.data_dir)
    nominal_path = root / "cube_nominal.h5"
    with h5py.File(nominal_path, "r") as ref:
        source_episode = ref["source_episode"][:]
        source_start = ref["source_start"][:]
        actions = ref["action"][:]
        qpos = ref["initial_qpos"][:]
        qvel = ref["initial_qvel"][:]
        block = ref["block_pos"][:]
        eff = ref["effector_pos"][:]
        contact = ref["gripper_contact"][:]
    result = {"groups": int(len(source_episode)), "variants": {}}
    for path in sorted(root.glob("cube_*.h5")):
        with h5py.File(path, "r") as f:
            name = str(f.attrs["variant"])
            checks = {
                "same_source_episode": bool(np.array_equal(f["source_episode"][:], source_episode)),
                "same_source_start": bool(np.array_equal(f["source_start"][:], source_start)),
                "same_initial_qpos": bool(np.array_equal(f["initial_qpos"][:], qpos)),
                "same_initial_qvel": bool(np.array_equal(f["initial_qvel"][:], qvel)),
                "same_actions": bool(np.array_equal(f["action"][:], actions)),
            }
            b = f["block_pos"][:]
            e = f["effector_pos"][:]
            c = f["gripper_contact"][:]
            checks.update({
                "final_block_delta_mean": float(np.linalg.norm(b[:, -1] - block[:, -1], axis=-1).mean()),
                "final_block_delta_max": float(np.linalg.norm(b[:, -1] - block[:, -1], axis=-1).max()),
                "final_effector_delta_mean": float(np.linalg.norm(e[:, -1] - eff[:, -1], axis=-1).mean()),
                "contact_delta_mean": float(np.abs(c - contact).mean()),
            })
            result["variants"][name] = checks
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
