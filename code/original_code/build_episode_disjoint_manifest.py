"""Select paired counterfactual starts with a strict source-episode split."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("manifests/cube_contact.npz"))
    ap.add_argument("--h5", type=Path, default=None,
                    help="sample from all valid source episodes when the manifest is too small")
    ap.add_argument("--out", type=Path, default=Path("manifests/cube_episode_disjoint.npz"))
    ap.add_argument("--train-groups", type=int, default=192)
    ap.add_argument("--test-groups", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.h5 is not None:
        with h5py.File(args.h5, "r") as f:
            ep_len = np.asarray(f["ep_len"][:], dtype=np.int64)
        valid = np.flatnonzero(ep_len >= 101)
        if len(valid) < args.train_groups + args.test_groups:
            raise ValueError("not enough valid episodes in source HDF5")
        episodes = valid.copy()
        by_episode = None
    else:
        src = np.load(args.source, allow_pickle=False)
        specs = np.asarray(src["train_specs"], dtype=np.int64)
        # Keep one start per source episode so the split is unambiguously episode-level.
        by_episode = {}
        for row in specs:
            by_episode.setdefault(int(row[0]), []).append(row)
        episodes = np.asarray(list(by_episode), dtype=np.int64)
        if len(episodes) < args.train_groups + args.test_groups:
            raise ValueError("manifest has too few unique episodes; pass --h5")
    rng.shuffle(episodes)

    test_eps = episodes[:args.test_groups]
    n_train_eps = min(args.train_groups, max(1, args.train_groups * 2 // 3))
    train_eps = episodes[args.test_groups:args.test_groups + n_train_eps]
    rows = []
    if by_episode is not None:
        rows.extend(by_episode[int(ep)][0] for ep in train_eps)
    else:
        with h5py.File(args.h5, "r") as f:
            ep_len = np.asarray(f["ep_len"][:], dtype=np.int64)
        for ep in train_eps:
            rows.append(np.asarray([ep, int(rng.integers(0, ep_len[ep] - 100))], dtype=np.int64))
    # Add extra starts only within train episodes to reach the requested group count.
    while len(rows) < args.train_groups:
        ep = int(train_eps[(len(rows) - len(train_eps)) % len(train_eps)])
        if by_episode is not None:
            candidates = by_episode[ep]
            rows.append(candidates[len(rows) % len(candidates)])
        else:
            with h5py.File(args.h5, "r") as f:
                ep_len = int(f["ep_len"][ep])
            rows.append(np.asarray([ep, int(rng.integers(0, ep_len - 100))], dtype=np.int64))
    if by_episode is not None:
        rows.extend(by_episode[int(ep)][0] for ep in test_eps)
    else:
        with h5py.File(args.h5, "r") as f:
            ep_len = np.asarray(f["ep_len"][:], dtype=np.int64)
        for ep in test_eps:
            rows.append(np.asarray([ep, int(rng.integers(0, ep_len[ep] - 100))], dtype=np.int64))
    selected = np.asarray(rows, dtype=np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        train_specs=selected,
        probe_tr=np.empty(0, dtype=np.int64),
        probe_val=np.empty(0, dtype=np.int64),
        train_group_end=np.asarray(args.train_groups, dtype=np.int64),
        test_group_start=np.asarray(args.train_groups, dtype=np.int64),
        seed=np.asarray(args.seed, dtype=np.int64),
    )
    print({
        "out": str(args.out),
        "rows": int(len(selected)),
        "train_groups": args.train_groups,
        "test_groups": args.test_groups,
        "train_episodes": int(len(set(train_eps.tolist()))),
        "test_episodes": int(len(set(test_eps.tolist()))),
        "overlap": int(len(set(train_eps.tolist()) & set(test_eps.tolist()))),
    })


if __name__ == "__main__":
    main()
