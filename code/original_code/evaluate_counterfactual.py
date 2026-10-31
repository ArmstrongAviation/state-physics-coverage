"""Evaluate LeWM one-step and free-running latent dynamics on paired Cube data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--files", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-groups", type=int, default=4)
    ap.add_argument("--group-start", type=int, default=0)
    ap.add_argument("--group-end", type=int, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    import train_data_arm as t

    device = torch.device(args.device)
    jepa = t.build_jepa_random(device)
    jepa.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    jepa.eval()

    result = {"checkpoint": str(args.checkpoint), "variants": {}}
    with torch.no_grad():
        for path in args.files:
            with h5py.File(path, "r") as f:
                total_groups, n_lat = f["pixels"].shape[:2]
                group_end = min(total_groups, args.group_end) if args.group_end is not None else total_groups
                if not 0 <= args.group_start < group_end:
                    raise ValueError(f"invalid group range [{args.group_start}, {group_end}) for {path}")
                n_groups = group_end - args.group_start
                teacher, rollout, teacher_cosine = [], [], []
                contact_teacher, free_teacher = [], []
                group_teacher = [[] for _ in range(n_groups)]
                group_rollout = [[] for _ in range(n_groups)]
                group_cosine = [[] for _ in range(n_groups)]
                for first in range(args.group_start, group_end, args.batch_groups):
                    last = min(group_end, first + args.batch_groups)
                    pix = np.asarray(f["pixels"][first:last], dtype=np.uint8)
                    raw = np.asarray(f["action"][first:last], dtype=np.float32)
                    contact = np.asarray(f["gripper_contact"][first:last], dtype=np.float32)
                    B = last - first
                    pixels, _, _ = t.to_gpu_normalized(pix, device)
                    cls = jepa.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
                    z = jepa.projector(cls).reshape(B, n_lat, t.EMB_DIM)
                    blocks = np.zeros((B, n_lat, 25), dtype=np.float32)
                    n_raw = raw.shape[1]
                    for k in range(n_lat):
                        segment = raw[:, k * 5:min((k + 1) * 5, n_raw)]
                        if segment.shape[1] < 5:
                            segment = np.concatenate([segment, np.zeros((B, 5 - segment.shape[1], 5), dtype=np.float32)], axis=1)
                        blocks[:, k] = segment.reshape(B, 25)
                    a = jepa.action_encoder(torch.from_numpy(blocks).to(device))
                    for ti in range(2, n_lat - 1):
                        pred = jepa.predict(z[:, ti - 2:ti + 1], a[:, ti - 2:ti + 1])[:, -1]
                        err = ((pred - z[:, ti + 1]) ** 2).mean(-1).cpu().numpy()
                        cos = (1.0 - F.cosine_similarity(pred.float(), z[:, ti + 1].float(), dim=-1)).cpu().numpy()
                        teacher.extend(err.tolist())
                        teacher_cosine.extend(cos.tolist())
                        for bi, (value, cosine_value) in enumerate(zip(err, cos)):
                            local_group = first - args.group_start + bi
                            group_teacher[local_group].append(float(value))
                            group_cosine[local_group].append(float(cosine_value))
                        flags = contact[:, ti + 1, 0] > 0.5
                        contact_teacher.extend(err[flags].tolist())
                        free_teacher.extend(err[~flags].tolist())
                    for bi in range(B):
                        hist = z[bi, :3].clone()
                        errs = []
                        for ti in range(2, n_lat - 1):
                            pred = jepa.predict(hist.unsqueeze(0), a[bi, ti - 2:ti + 1].unsqueeze(0))[0, -1]
                            errs.append(float(((pred - z[bi, ti + 1]) ** 2).mean().cpu()))
                            hist = torch.cat([hist[1:], pred.unsqueeze(0)], dim=0)
                        rollout.extend(errs)
                        group_rollout[first - args.group_start + bi].extend(errs)
                name = str(f.attrs.get("variant", path.stem))
                result["variants"][name] = {
                    "groups": int(n_groups),
                    "group_start": args.group_start,
                    "group_end": group_end,
                    "teacher_mse": float(np.mean(teacher)),
                    "teacher_p95": float(np.percentile(teacher, 95)),
                    "teacher_cosine_error": float(np.mean(teacher_cosine)),
                    "rollout_mse": float(np.mean(rollout)),
                    "contact_teacher_mse": float(np.mean(contact_teacher)) if contact_teacher else None,
                    "free_teacher_mse": float(np.mean(free_teacher)) if free_teacher else None,
                    "n_teacher": len(teacher),
                    "group_teacher_mse": [float(np.mean(x)) for x in group_teacher],
                    "group_teacher_cosine_error": [float(np.mean(x)) for x in group_cosine],
                    "group_rollout_mse": [float(np.mean(x)) for x in group_rollout],
                }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
