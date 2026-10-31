"""Verify that a paired counterfactual split has no source-episode overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--train-end", type=int, default=192)
    ap.add_argument("--test-start", type=int, default=192)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with h5py.File(args.data, "r") as f:
        episodes = f["source_episode"][:]
        starts = f["source_start"][:]
    train = set(int(x) for x in episodes[:args.train_end])
    test = set(int(x) for x in episodes[args.test_start:])
    result = {
        "groups": int(len(episodes)),
        "train_groups": args.train_end,
        "test_groups": int(len(episodes) - args.test_start),
        "train_episodes": len(train),
        "test_episodes": len(test),
        "episode_overlap": sorted(train & test),
        "episode_overlap_count": len(train & test),
        "pair_overlap_count": len(
            set(zip(episodes[:args.train_end].tolist(), starts[:args.train_end].tolist()))
            & set(zip(episodes[args.test_start:].tolist(), starts[args.test_start:].tolist()))
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if result["episode_overlap_count"] or result["pair_overlap_count"]:
        raise SystemExit("episode-disjoint audit failed")


if __name__ == "__main__":
    main()
