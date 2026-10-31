# Minimal revision protocol — frozen before d11 training

Purpose: add the missing independent-physics collection baseline and physically interpretable multi-step evaluation without retraining existing baselines. No test-driven checkpoint, seed, or endpoint selection.

## Training

- State: reuse d8_cf_state_s0, d8_cf_state_s1, d10_cf_state_s3. The stored training arguments match except seed/tag. Data: nominal groups 0–191.
- Paired: reuse d8_cf_mechanism_s0/s1/s2, nominal/mass_high/delay_2 groups 0–63.
- Unpaired: d11_cf_unpaired_s0/s1/s2; use the existing simulated pool, partition groups 0–191 into three disjoint sets of 64, one set per physical condition. The selection seed is 20261030, shared across all model seeds. No simulator rerun or label shuffling is needed: the selected trajectories actually have different initial-state/action sources across conditions.
- All arms: 384 window records, 10 latent frames/window, latent starts 0 and 10, 10,000 AdamW updates, batch size 32, learning rate 5e-5, weight decay 1e-3, warmup 2,000 steps, cosine decay, SIGReg weight 0.09, BF16, CEReg off. Existing trainer/model/loss are unchanged.
- The independent collection arm necessarily covers more source states/actions than paired reuse. This tests collection strategies under a fixed budget; it does not isolate a pairing effect while holding unique state coverage fixed. The trainer has no pair-specific objective.

## Existing-data audit correction

The state pool has 192 group records but only 191 distinct (source episode, source start) pairs: records 45 and 173 both use episode 4022, start 97. Their initial qpos/qvel, actions and block-position trajectories match, while pixel arrays differ. Preserve the original baseline and disclose the distinction between distinct cache records and distinct dynamical sources. The unpaired assignment keeps these duplicate source records in the same physics condition; if the initial random partition separates them, exchange record 173 with the first non-45 record in record 45's partition. This preserves exact physical-condition counts and prevents cross-condition reuse of that source. The actual source-window identity count is saved in data_audit.json. Training and test episodes are disjoint.

## Physical evaluation

- Freeze all world models. Use exactly the same nominal training groups 0–191 and frames 0–19 for every model's calibration (3,840 frames). This common labeled calibration set is evaluation-only; it is deliberately independent of which groups the world model saw. No composed-condition calibration or test-group fitting.
- Fit a linear ridge decoder from projected latent (192 dimensions) to block xyz and effector xyz. Select alpha from [1e-5,1e-4,1e-3,1e-2,1e-1,1,10] using a common 80/20 split of the 128 training source episodes, seeded 20261030. Duplicate starts/episodes remain in the same calibration split. Standardization uses fitting rows only. Select by six-coordinate normalized validation MSE, then refit on all calibration rows. Save all validation scores and fitted decoder parameters.
- Test groups 192–255 (64 distinct episodes) under nominal, mass_high, delay_2, mass_high_delay_2; the last is the primary held-out composition.
- Initialize history with true frames 0,1,2, then recursively predict frames 3–20: 18 future latent steps, each spanning 5 raw actions, for a 90-action forecast after the last observed frame. Physical duration in seconds is not asserted until environment timing is verified.
- Primary physical endpoint: mean block Euclidean position error in meters over all 64 groups and 18 future frames. Also save per-horizon/per-group arrays, effector error, coordinate MSE, teacher-forced decoding error, and decoding error from observed future latent vectors.
- References: hold the true state at frame 2 constant; hold the decoded frame-2 latent constant. These distinguish motion prediction from static position decoding. They are reference baselines, not learned controllers.
- Recompute raw latent teacher/rollout MSE and cosine error and compare per-group values to the original evaluator with a 2e-4 absolute tolerance. Also report rollout MSE standardized by calibration latent standard deviations. A linear decoder remains an imperfect measurement; physical gains are not assumed and no control success is claimed.

## Statistical reporting

Three model seeds per family; state seeds are 0,1,3 and other seeds 0,1,2. Report mean, sample seed SD, and individual seed results. Pair only the common test episodes, not nominally equal training seed labels. Produce 10,000-replicate conditional group-only intervals (average models first) and seed+group intervals (independently resample trained seeds within each family, and resample common test episodes jointly). Three seeds are a limited sample and these intervals are sensitivity estimates. Report all directions and all four test variants; no selection based on significance.

## Execution

Use GPUs 0,2,3 for the three new training jobs and GPU 4 for existing-checkpoint physical evaluation. GPU 1 stays reserved per the existing project instructions; GPU 5 is already occupied. Each new job proceeds to original latent evaluation then physical evaluation. The supervisor runs summary generation only if every worker completes successfully, and records failures in status/*.json. All additions live in revision_minimal_20261030, cache/cf_v3_unpaired_d11 and exp5b_runs/d11_cf_unpaired_s*. No original results or source files are overwritten.
