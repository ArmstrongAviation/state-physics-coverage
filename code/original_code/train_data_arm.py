"""
exp5b_cereg_train.py
LeWM-CEReg 5b: 从头完整训练（random init），与 LeWM 原版可对照。

5a 推荐超参数（路径一）：w=10, percentile=0.85
匹配原版 LeWM 训练 config（config/train/lewm.yaml）：
  - lr = 5e-5  (AdamW + LinearWarmupCosineAnnealingLR)
  - weight_decay = 1e-3
  - SIGReg λ = 0.09
  - precision = bf16

三件监控（user 要求）：
  1. 训练稳定性：每 1k steps 打 L/L_pred/L_sig/top10%MSE
  2. PushT straightness：每 20k steps 测一次，baseline ≈ 0.72，红线 0.60
  3. contact/free MSE ratio：每 eval-every steps 测，绘训练曲线

用法：
  python exp5b_cereg_train.py --steps 100000 --batch-size 64 --eval-every 5000
  python exp5b_cereg_train.py --steps 50000 --no-cereg   # baseline 对照（不加 CEReg，只 SIGReg）
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms

REPO_ROOT = Path(__file__).parent
SOURCE_REPO = Path(os.environ.get("LEWM_SOURCE_REPO", "/data/chenyizi/le-wm"))
CKPT_DIR  = SOURCE_REPO / "checkpoints"
DATA_DIR  = SOURCE_REPO / "datasets"
SP_PATH   = SOURCE_REPO / ".venv/lib/python3.10/site-packages/stable_pretraining"

sys.path.insert(0, str(SOURCE_REPO))

# ── constants (match train.yaml + cube data) ──
FRAMESKIP    = 5
HISTORY_SIZE = 3
EMB_DIM      = 192
ACT_DIM      = 5
ACT_BLOCK    = FRAMESKIP * ACT_DIM   # 25
SEQ_LAT      = 10
H5_PATH      = DATA_DIR / "lewm-cube" / "cube_single_expert.h5"

# PushT for straightness eval
PUSHT_H5     = DATA_DIR / "lewm-pusht" / "pusht_expert_train.h5"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── model construction (random init, mirroring train.py) ──
def _load_sp_backbone():
    for n, r in [
        ("stable_pretraining.backbone",       "backbone/__init__.py"),
        ("stable_pretraining.backbone.utils", "backbone/utils.py"),
    ]:
        if n not in sys.modules:
            spec = importlib.util.spec_from_file_location(n, SP_PATH / r)
            mod  = importlib.util.module_from_spec(spec)
            sys.modules[n] = mod
            spec.loader.exec_module(mod)


def build_jepa_random(device, hidden_dim=192, embed_dim=192,
                     img_size=224, patch_size=14):
    """从头构造 JEPA — 随机初始化（匹配 train.py 第 82-124 行）."""
    _load_sp_backbone()
    from stable_pretraining.backbone.utils import vit_hf
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder = vit_hf("tiny", patch_size=patch_size, image_size=img_size,
                     pretrained=False, use_mask_token=False)

    predictor = ARPredictor(
        num_frames=HISTORY_SIZE,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=6, heads=16, mlp_dim=2048, dim_head=64,
        dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=ACT_BLOCK, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                         hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)

    return JEPA(encoder=encoder, predictor=predictor,
                action_encoder=action_encoder,
                projector=projector, pred_proj=predictor_proj).to(device)


def import_sigreg():
    _load_sp_backbone()
    spec = importlib.util.spec_from_file_location(
        "stable_pretraining.methods.lejepa", SP_PATH / "methods" / "lejepa.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stable_pretraining.methods.lejepa"] = mod
    spec.loader.exec_module(mod)
    return mod.SlicedEppsPulley


# ── dataset (re-use from 5a fast) ──
class CubeWindowDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, ep_ids, seq_lat=SEQ_LAT, frameskip=FRAMESKIP,
                 act_dim=ACT_DIM, window_specs=None, cache_dir=None):
        self.h5_path = h5_path
        self.ep_ids = ep_ids
        self.seq_lat = seq_lat
        self.fs = frameskip
        self.act_dim = act_dim
        with h5py.File(h5_path, "r") as f:
            self.ep_offset = f["ep_offset"][:]
            self.ep_len    = f["ep_len"][:]
        self.min_raw = seq_lat * frameskip
        self._h5 = None
        self.window_specs = None if window_specs is None else np.asarray(window_specs, dtype=np.int64)
        self.cache_pixels = self.cache_actions = None
        if cache_dir:
            cache_dir = Path(cache_dir)
            self.cache_pixels = np.load(cache_dir / "pixels.npy", mmap_mode="r")
            self.cache_actions = np.load(cache_dir / "actions.npy", mmap_mode="r")

    def _open(self):
        import os
        pid = os.getpid()
        if self._h5 is None or getattr(self, "_h5_pid", None) != pid:
            self._h5 = h5py.File(self.h5_path, "r")
            self._h5_pid = pid
        return self._h5

    def __len__(self):
        return len(self.window_specs) if self.window_specs is not None else len(self.ep_ids) * 10

    def __getitem__(self, idx):
        if self.cache_pixels is not None:
            return np.asarray(self.cache_pixels[idx]), np.asarray(self.cache_actions[idx], dtype=np.float32)
        f = self._open()
        for _retry in range(10):
            if self.window_specs is not None:
                ep = int(self.window_specs[idx, 0])
                start = int(self.window_specs[idx, 1])
            else:
                ep = int(self.ep_ids[(idx + _retry) % len(self.ep_ids)])
                start = None
            off = int(self.ep_offset[ep])
            L   = int(self.ep_len[ep])
            max_start = max(0, L - self.min_raw)
            if start is None:
                start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            start = min(start, max_start)
            lat_idx = start + np.arange(self.seq_lat) * self.fs
            raw = f["action"][off + start: off + start + self.min_raw]
            if np.isnan(raw).any():
                continue
            pix = f["pixels"][off + lat_idx]
            acts = raw.reshape(self.seq_lat, self.fs * self.act_dim)
            return pix, acts.astype(np.float32)
        pix = f["pixels"][off + lat_idx]
        acts = np.nan_to_num(raw.reshape(self.seq_lat, self.fs * self.act_dim))
        return pix, acts.astype(np.float32)


def collate(batch):
    pix = np.stack([b[0] for b in batch])
    act = np.stack([b[1] for b in batch])
    return pix, torch.from_numpy(act)


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    if info is not None:
        np.random.seed((info.seed + worker_id) % (2**32))


# ── pixels → normalized GPU tensor (no torchvision per-frame) ──
_IMG_MEAN_GPU = None
_IMG_STD_GPU  = None
def to_gpu_normalized(pix_np, device):
    global _IMG_MEAN_GPU, _IMG_STD_GPU
    B, T, H, W, C = pix_np.shape
    t = torch.from_numpy(pix_np).to(device, non_blocking=True)
    t = t.reshape(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
    t = t.float().div_(255.0)
    if _IMG_MEAN_GPU is None or _IMG_MEAN_GPU.device != t.device:
        _IMG_MEAN_GPU = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        _IMG_STD_GPU  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)
    return (t - _IMG_MEAN_GPU) / _IMG_STD_GPU, B, T


# ── monitoring: PushT straightness ──
@torch.no_grad()
def eval_pusht_straightness(jepa, n_eps=100, device="cuda", batch_size=64):
    """Return mean straightness on PushT (baseline ≈ 0.72)."""
    if not PUSHT_H5.exists():
        return float("nan")
    transform_fn = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    jepa.eval()
    rng = np.random.default_rng(42)
    with h5py.File(PUSHT_H5, "r") as f:
        ep_offset = f["ep_offset"][:]; ep_len = f["ep_len"][:]
        valid = np.where(ep_len >= 5 * 10)[0]
        chosen = rng.choice(valid, size=min(n_eps, len(valid)), replace=False)
        scores = []
        for ep_id in chosen:
            off = int(ep_offset[ep_id]); L = int(ep_len[ep_id])
            lat_idx = np.arange(0, L, FRAMESKIP)
            if len(lat_idx) < 3:
                continue
            pix = f["pixels"][off + lat_idx]
            embs = []
            for s in range(0, len(pix), batch_size):
                t = torch.stack([transform_fn(im) for im in pix[s:s+batch_size]]).to(device)
                cls = jepa.encoder(t, interpolate_pos_encoding=True).last_hidden_state[:, 0]
                embs.append(jepa.projector(cls).cpu())
            z = torch.cat(embs, 0).numpy()
            dz = np.diff(z, axis=0)
            dz = dz / (np.linalg.norm(dz, axis=1, keepdims=True) + 1e-9)
            cos = (dz[:-1] * dz[1:]).sum(-1)
            scores.append(float(cos.mean()))
    return float(np.mean(scores)) if scores else float("nan")


# ── monitoring: contact/free predictor MSE ──
@torch.no_grad()
def eval_contact_mse(jepa, h5_path, ep_ids, device, n_eps=100, batch_size=64):
    from exp4_contact_mse import detect_contact_events
    transform_fn = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    jepa.eval()
    contact_e, free_e = [], []
    with h5py.File(h5_path, "r") as f:
        ep_offset = f["ep_offset"][:]; ep_len = f["ep_len"][:]
        for ep_id in ep_ids[:n_eps]:
            off = int(ep_offset[ep_id]); L = int(ep_len[ep_id])
            lat_idx = np.arange(0, L, FRAMESKIP); T_lat = len(lat_idx)
            if T_lat < HISTORY_SIZE + 2:
                continue
            pix = f["pixels"][off + lat_idx]
            embs = []
            for s in range(0, len(pix), batch_size):
                t = torch.stack([transform_fn(im) for im in pix[s:s+batch_size]]).to(device)
                cls = jepa.encoder(t, interpolate_pos_encoding=True).last_hidden_state[:, 0]
                embs.append(jepa.projector(cls))
            z_all = torch.cat(embs, 0)

            raw = f["action"][off: off + L].astype(np.float32)
            raw = np.nan_to_num(raw, 0.0)
            ab = np.zeros((T_lat, ACT_BLOCK), dtype=np.float32)
            for i, s in enumerate(lat_idx):
                blk = raw[s:s+FRAMESKIP]
                if len(blk) < FRAMESKIP:
                    blk = np.concatenate([blk, np.zeros((FRAMESKIP - len(blk), ACT_DIM), dtype=np.float32)])
                ab[i] = blk.flatten()
            act_t = torch.from_numpy(ab).to(device)

            ep_dict = {
                "gripper_contact": f["proprio_gripper_contact"][off:off+L, 0],
                "gripper_opening": f["proprio_gripper_opening"][off:off+L, 0],
                "block_z":         f["privileged_block_0_pos"][off:off+L, 2],
            }
            _, event_lat, _ = detect_contact_events(ep_dict, frameskip=FRAMESKIP)
            n_lat_e = len(event_lat)
            for t in range(HISTORY_SIZE - 1, min(T_lat - 1, n_lat_e - 1)):
                t0 = t - HISTORY_SIZE + 1
                z_h = z_all[t0:t+1].unsqueeze(0)
                a_h = jepa.action_encoder(act_t[t0:t+1].unsqueeze(0))
                p = jepa.predict(z_h, a_h)[0, -1]
                mse = ((p - z_all[t+1]) ** 2).mean().item()
                if event_lat[t+1]:
                    contact_e.append(mse)
                else:
                    free_e.append(mse)
    if not contact_e or not free_e:
        return None
    c = np.array(contact_e); f_ = np.array(free_e)
    return {
        "ratio":        float(c.mean() / f_.mean()),
        "contact_mean": float(c.mean()),
        "free_mean":    float(f_.mean()),
        "contact_p95":  float(np.percentile(c, 95)),
        "n_contact":    len(c),
        "n_free":       len(f_),
    }


# ── monitoring: state probes ──
@torch.no_grad()
def eval_state_probes(jepa, h5_path, train_eps, val_eps, device):
    transform_fn = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    jepa.eval()
    def encode(eps):
        embs, grip, block, ee = [], [], [], []
        with h5py.File(h5_path, "r") as f:
            ep_offset = f["ep_offset"][:]; ep_len = f["ep_len"][:]
            for ep in eps:
                off = int(ep_offset[ep]); L = int(ep_len[ep])
                lat_idx = np.arange(0, L, FRAMESKIP)
                pix = f["pixels"][off + lat_idx]
                e = []
                for s in range(0, len(pix), 64):
                    t = torch.stack([transform_fn(im) for im in pix[s:s+64]]).to(device)
                    cls = jepa.encoder(t, interpolate_pos_encoding=True).last_hidden_state[:, 0]
                    e.append(jepa.projector(cls).cpu().numpy())
                embs.append(np.concatenate(e, 0))
                grip.append(f["proprio_gripper_opening"][off + lat_idx])
                block.append(f["privileged_block_0_pos"][off + lat_idx])
                ee.append(f["proprio_effector_pos"][off + lat_idx])
        return (np.concatenate(embs).astype(np.float32),
                np.concatenate(grip).astype(np.float32),
                np.concatenate(block).astype(np.float32),
                np.concatenate(ee).astype(np.float32))

    def fit_probe(X, y):
        sm = y.mean(0); ss = y.std(0) + 1e-8
        yn = (y - sm) / ss
        A = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)
        Wf, *_ = np.linalg.lstsq(A, yn, rcond=None)
        return Wf[:-1], Wf[-1], sm, ss
    def probe_r(emb, tgt, W, b, sm, ss):
        pr = emb @ W + b
        tn = (tgt - sm) / ss
        return float(np.mean([np.corrcoef(pr[:, k], tn[:, k])[0, 1] for k in range(tgt.shape[1])]))

    Xtr, gtr, btr, etr = encode(train_eps)
    Xv, gv, bv, ev_ = encode(val_eps)
    res = {}
    for name, ytr, yv in [("gripper", gtr, gv), ("block_pos", btr, bv), ("ee_pos", etr, ev_)]:
        W, b, sm, ss = fit_probe(Xtr, ytr)
        res[name] = probe_r(Xv, yv, W, b, sm, ss)
    return res


# ── training step ──
def train_step(jepa, sigreg, batch, device, w, percentile, lamb, use_cereg, use_amp):
    pix_np, acts = batch
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        pixels, B, T = to_gpu_normalized(pix_np, device)
        acts = acts.to(device)

        cls = jepa.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        z = jepa.projector(cls).reshape(B, T, EMB_DIM)
        act_emb = jepa.action_encoder(acts)

        HS = HISTORY_SIZE
        z_targets, z_preds = [], []
        for t in range(HS - 1, T - 1):
            pred = jepa.predict(z[:, t-HS+1:t+1], act_emb[:, t-HS+1:t+1])
            z_preds.append(pred[:, -1])
            z_targets.append(z[:, t+1])
        z_pred = torch.stack(z_preds, dim=1)                     # (B, T-HS, D)
        z_tgt  = torch.stack(z_targets, dim=1)

        per_step = ((z_pred - z_tgt.detach()) ** 2).mean(-1)     # (B, T-HS)
        flat = per_step.reshape(-1)

        if use_cereg:
            threshold = torch.quantile(flat.detach().float(), percentile)
            hard_mask = (flat > threshold).float()
            weights = 1.0 + (w - 1.0) * hard_mask
            L_pred = (flat * weights).mean()
        else:
            L_pred = flat.mean()

        z_flat = z.reshape(-1, EMB_DIM)
        L_sig = sigreg(z_flat)
        L = L_pred + lamb * L_sig

    return L, L_pred.detach(), L_sig.detach(), per_step.detach()


# ── main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w",          type=float, default=10.0)
    parser.add_argument("--percentile", type=float, default=0.85)
    parser.add_argument("--lamb",       type=float, default=0.09,    help="match train.yaml SIGReg weight")
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--weight-decay",type=float,default=1e-3)
    parser.add_argument("--steps",      type=int,   default=100000)
    parser.add_argument("--warmup-steps",type=int,  default=2000)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-train-eps",type=int,   default=8000)
    parser.add_argument("--n-probe-tr", type=int,   default=300)
    parser.add_argument("--n-probe-val",type=int,   default=200)
    parser.add_argument("--num-workers",type=int,   default=6)
    parser.add_argument("--no-cereg",   action="store_true",
                        help="disable CEReg (= original LeWM training, baseline)")
    parser.add_argument("--no-amp",     action="store_true", help="disable bf16 autocast")
    parser.add_argument("--log-every",  type=int,   default=1000,
                        help="log L/L_pred/L_sig every N steps (user req: 1k)")
    parser.add_argument("--eval-every", type=int,   default=5000,
                        help="contact/free MSE eval every N steps")
    parser.add_argument("--straight-every",type=int,default=20000,
                        help="PushT straightness eval every N steps (user req: 20k)")
    parser.add_argument("--save-every", type=int,   default=20000)
    parser.add_argument("--resume-from", default=None,
                        help="path to weights_stepXXX.pt to resume from")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="step counter to resume from (must match weights filename)")
    parser.add_argument("--tag",        default=None)
    parser.add_argument("--window-manifest", default=None,
                        help="npz with train_specs (N,2) and probe_eps arrays")
    parser.add_argument("--cache-dir", default=None,
                        help="memmap cache with pixels.npy/actions.npy")
    parser.add_argument("--final-contact-eps", type=int, default=50)
    parser.add_argument("--final-straight-eps", type=int, default=50)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--skip-initial-eval", action="store_true")
    args = parser.parse_args()

    tag = args.tag or ("baseline" if args.no_cereg else f"cereg_w{args.w}_p{args.percentile}")
    out_dir = REPO_ROOT / "exp5b_runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"
    metrics_path = out_dir / "metrics.json"

    metrics = {"train": [], "contact_mse": [], "straightness": [], "probes": [], "args": vars(args)}

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as fp: fp.write(msg + "\n")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    log(f"[5b] tag={tag}  CEReg={'OFF' if args.no_cereg else 'ON'}  "
        f"w={args.w}  p={args.percentile}  λ={args.lamb}  lr={args.lr}")
    log(f"     steps={args.steps}  bs={args.batch_size}  wd={args.weight_decay}  "
        f"warmup={args.warmup_steps}  amp={'OFF' if args.no_amp else 'BF16'}")

    log("Building random-init JEPA …")
    jepa = build_jepa_random(device)
    SP = import_sigreg()
    sigreg = SP(num_slices=1024).to(device)        # match train.yaml num_proj=1024

    if args.resume_from:
        log(f"Resuming weights from {args.resume_from} (resume_step={args.resume_step})")
        sd = torch.load(args.resume_from, map_location=device, weights_only=True)
        jepa.load_state_dict(sd)

    with h5py.File(H5_PATH, "r") as f:
        ep_len = f["ep_len"][:]
    valid = np.where(ep_len >= SEQ_LAT * FRAMESKIP)[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(valid)
    train_eps = perm[:args.n_train_eps]
    probe_tr  = perm[args.n_train_eps : args.n_train_eps + args.n_probe_tr]
    probe_val = perm[args.n_train_eps + args.n_probe_tr : args.n_train_eps + args.n_probe_tr + args.n_probe_val]
    train_specs = None
    if args.window_manifest:
        manifest = np.load(args.window_manifest, allow_pickle=False)
        train_specs = manifest["train_specs"]
        if "probe_tr" in manifest:
            probe_tr = manifest["probe_tr"]
        if "probe_val" in manifest:
            probe_val = manifest["probe_val"]
        train_eps = np.unique(train_specs[:, 0])
        log(f"     window_manifest={args.window_manifest}  train_windows={len(train_specs)}")
    log(f"     train_eps={len(train_eps)}  probe_tr={len(probe_tr)}  probe_val={len(probe_val)}")

    ds = CubeWindowDataset(H5_PATH, train_eps, window_specs=train_specs,
                           cache_dir=args.cache_dir)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, collate_fn=collate,
        worker_init_fn=worker_init_fn,
        persistent_workers=(args.num_workers > 0), pin_memory=True,
    )

    opt = torch.optim.AdamW(jepa.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # LinearWarmup + Cosine
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # if resuming, fast-forward LR scheduler to resume_step
    if args.resume_step > 0:
        for _ in range(args.resume_step):
            sched.step()
        log(f"  fast-forwarded LR scheduler to step {args.resume_step}; "
            f"lr={opt.param_groups[0]['lr']:.2e}")

    use_cereg = not args.no_cereg
    use_amp   = not args.no_amp

    if args.resume_step == 0 and not args.skip_initial_eval:
        log("Initial measurements (random init) …")
        init_str = eval_pusht_straightness(jepa, n_eps=50, device=device)
        log(f"  init PushT straightness = {init_str:.4f}")

    jepa.train()
    step = args.resume_step
    t0 = time.time()
    log_buf = []
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            L, L_pred, L_sig, per_step = train_step(
                jepa, sigreg, batch, device,
                w=args.w, percentile=args.percentile, lamb=args.lamb,
                use_cereg=use_cereg, use_amp=use_amp)

            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(jepa.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1

            # train-step log
            log_buf.append((step, L.item(), L_pred.item(), L_sig.item(),
                            per_step.flatten().topk(max(1, per_step.numel()//10))[0].mean().item()))

            if step % args.log_every == 0:
                last = log_buf[-1]
                # mean over last log_every steps
                recent = log_buf[-args.log_every:]
                L_avg = np.mean([r[1] for r in recent])
                Lp_avg = np.mean([r[2] for r in recent])
                Ls_avg = np.mean([r[3] for r in recent])
                top10 = np.mean([r[4] for r in recent])
                lr_now = opt.param_groups[0]["lr"]
                log(f"  step {step:6d}/{args.steps}  L={L_avg:.4f}  "
                    f"L_pred={Lp_avg:.4f}  L_sig={Ls_avg:.4f}  top10%MSE={top10:.4f}  "
                    f"lr={lr_now:.2e}  elapsed={time.time()-t0:.0f}s")
                metrics["train"].append({
                    "step": step, "L": float(L_avg), "L_pred": float(Lp_avg),
                    "L_sig": float(Ls_avg), "top10_mse": float(top10), "lr": lr_now,
                })

            if step % args.eval_every == 0:
                jepa.eval()
                cm = eval_contact_mse(jepa, H5_PATH, probe_val, device, n_eps=50)
                if cm is not None:
                    log(f"  step {step:6d}  RATIO  contact/free = {cm['ratio']:.3f}×  "
                        f"(c={cm['contact_mean']:.4f} f={cm['free_mean']:.4f}  "
                        f"c_p95={cm['contact_p95']:.4f}  n_c={cm['n_contact']} n_f={cm['n_free']})")
                    metrics["contact_mse"].append({"step": step, **cm})
                jepa.train()

            if step % args.straight_every == 0:
                jepa.eval()
                s = eval_pusht_straightness(jepa, n_eps=100, device=device)
                log(f"  step {step:6d}  PushT straightness = {s:.4f}  (LeWM final cube ckpt ≈ 0.68)")
                metrics["straightness"].append({"step": step, "value": s})
                # No warning: from-scratch training doesn't reach 0.7+ in 100k steps
                # (LeWM official trains 100 epochs × bs128 ≈ 600k effective steps).
                # We compare cereg vs baseline trajectory, not absolute value.
                jepa.train()

            if step % args.save_every == 0:
                ckpt_path = out_dir / f"weights_step{step}.pt"
                torch.save(jepa.state_dict(), ckpt_path)
                log(f"  step {step:6d}  saved → {ckpt_path}")
                with open(metrics_path, "w") as fp:
                    json.dump(metrics, fp, indent=2)

    # ── final eval ──
    if args.skip_final_eval:
        torch.save(jepa.state_dict(), out_dir / "weights_final.pt")
        with open(metrics_path, "w") as fp:
            json.dump(metrics, fp, indent=2)
        log(f"  weights → {out_dir / 'weights_final.pt'}")
        return
    jepa.eval()
    log(f"\n[FINAL] tag={tag}  total_steps={step}  elapsed={time.time()-t0:.0f}s")
    cm = eval_contact_mse(jepa, H5_PATH, probe_val, device, n_eps=args.final_contact_eps)
    if cm is not None:
        log(f"  contact/free ratio = {cm['ratio']:.3f}×")
        log(f"  contact_mean = {cm['contact_mean']:.4f}  free_mean = {cm['free_mean']:.4f}")
        log(f"  contact_p95  = {cm['contact_p95']:.4f}")
        metrics["final_contact_mse"] = cm
    s = eval_pusht_straightness(jepa, n_eps=args.final_straight_eps, device=device)
    log(f"  PushT straightness = {s:.4f}")
    metrics["final_straightness"] = s

    pr = eval_state_probes(jepa, H5_PATH, probe_tr, probe_val, device)
    log(f"  state probes: gripper={pr['gripper']:.4f}  block={pr['block_pos']:.4f}  ee={pr['ee_pos']:.4f}")
    metrics["final_probes"] = pr

    torch.save(jepa.state_dict(), out_dir / "weights_final.pt")
    with open(metrics_path, "w") as fp:
        json.dump(metrics, fp, indent=2)
    log(f"  weights → {out_dir / 'weights_final.pt'}")
    log(f"  metrics → {metrics_path}")


if __name__ == "__main__":
    main()
