"""Materialize paired-counterfactual HDF5 groups into LeWM training windows.

The resulting cache follows the same ``pixels.npy``/``actions.npy`` contract as
the expert-data cache, so the existing LeWM trainer can compare data arms
without changing the model or loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--starts", default="0,10", help="latent starts for each 21-frame group")
    ap.add_argument("--max-groups", type=int, default=None)
    ap.add_argument("--group-start", type=int, default=0)
    ap.add_argument("--group-end", type=int, default=None)
    ap.add_argument("--repeat-to", type=int, default=None,
                     help="repeat/sample windows to this fixed frame budget")
    args = ap.parse_args()
    starts = [int(x) for x in args.starts.split(",") if x.strip()]
    if not starts:
        raise ValueError("at least one latent start is required")

    entries = []
    variants = []
    for path in args.files:
        with h5py.File(path, "r") as f:
            n_groups, n_lat = f["pixels"].shape[:2]
            n_groups = min(n_groups, args.max_groups) if args.max_groups else n_groups
            group_end = min(n_groups, args.group_end) if args.group_end is not None else n_groups
            if not 0 <= args.group_start < group_end:
                raise ValueError(f"invalid group range [{args.group_start}, {group_end}) for {path}")
            variant = str(f.attrs.get("variant", path.stem))
            for group in range(args.group_start, group_end):
                for start in starts:
                    if start + 10 <= n_lat:
                        entries.append((str(path), group, start))
                        variants.append(variant)
    if not entries:
        raise ValueError("no valid windows")

    rng = np.random.default_rng(2026)
    order = np.arange(len(entries))
    if args.repeat_to is not None:
        if args.repeat_to <= 0:
            raise ValueError("repeat-to must be positive")
        order = rng.choice(order, size=args.repeat_to, replace=len(order) < args.repeat_to)
    else:
        rng.shuffle(order)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    n = len(order)
    pixels = np.lib.format.open_memmap(out / "pixels.npy", mode="w+", dtype=np.uint8,
                                       shape=(n, 10, 224, 224, 3))
    actions = np.lib.format.open_memmap(out / "actions.npy", mode="w+", dtype=np.float32,
                                        shape=(n, 10, 25))
    labels = np.empty(n, dtype="U64")
    source_group = np.empty((n, 2), dtype=np.int64)

    handles = {}
    try:
        for i, original_idx in enumerate(order):
            path_str, group, start = entries[int(original_idx)]
            if path_str not in handles:
                handles[path_str] = h5py.File(path_str, "r")
            f = handles[path_str]
            pixels[i] = f["pixels"][group, start:start + 10]
            raw = np.asarray(f["action"][group, start * 5:(start + 10) * 5], dtype=np.float32)
            actions[i] = np.nan_to_num(raw).reshape(10, 25)
            labels[i] = variants[int(original_idx)]
            source_group[i] = (group, start)
            if (i + 1) % 128 == 0:
                pixels.flush(); actions.flush()
                print(f"{i + 1}/{n}", flush=True)
    finally:
        for f in handles.values():
            f.close()
    pixels.flush(); actions.flush()
    np.save(out / "labels.npy", labels)
    np.save(out / "source_group.npy", source_group)
    specs = np.column_stack([np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    np.savez_compressed(out / "manifest.npz", train_specs=specs,
                        probe_tr=np.empty(0, dtype=np.int64),
                        probe_val=np.empty(0, dtype=np.int64),
                        variants=np.unique(labels))
    metadata = {
        "files": [str(p) for p in args.files],
        "starts": starts,
        "group_start": args.group_start,
        "group_end": args.group_end,
        "windows": n,
        "unique_windows": len(entries),
        "repeat_to": args.repeat_to,
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
