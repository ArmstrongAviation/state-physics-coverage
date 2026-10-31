SUPPLEMENTARY CODE AND NUMERICAL DATA
State and Physics Coverage for Visual World Models:
A Fixed-Budget Empirical Study
Prepared 2026-10-31
Correspondence: Weichang Li, aumax@mail.scut.edu.cn

1. SCOPE
This archive supports independent reaggregation of the main experimental
results, reconstruction of the manuscript tables/figures, and auditing of
source splits and replay states/actions. It also includes the original
training, simulator-generation, and evaluation scripts for inspection.

It is NOT a self-contained package for retraining or rerunning visual
inference. The original expert HDF5, full rendered replay HDF5 files, and
world-model checkpoints are not included. The four replay files with images
occupy approximately 550 MB in total. No claim of public hosting or automatic
availability of the omitted files is made. See DATA_DICTIONARY.txt and
FULL_EXPERIMENT_REQUIREMENTS.txt for the precise boundaries.

2. QUICK START: CPU-ONLY NUMERICAL REPRODUCTION
Python 3.10 or later and NumPy are sufficient for this step.

  python -m pip install -r requirements-analysis.txt
  python tools/reproduce_and_verify.py

The script reaggregates all nine models over four test conditions with the
original 10,000-replicate bootstrap, compares numerical results against the
manuscript reference, checks the non-image replay data and persistence
errors, and writes verification_report.json. It does not train models,
download data, change checkpoints, or require the original server paths.
Numerical tolerance is stated in the verification report.

3. REBUILD THE MANUSCRIPT TABLES AND FIGURES

  python -m pip install -r requirements-figures.txt
  python manuscript_assets/scripts/build_tables_figures.py

Outputs are written to manuscript_assets/tables and manuscript_assets/figures.
The framework illustration uses two unchanged robot views extracted from the
authors' original diagram; they illustrate the task, not measured variant
trajectories. Main result tables use the archived numerical reports.
The full paper and LaTeX submission package are separate files.

4. CONTENTS
revision_minimal_20261030/
  Original added scripts, frozen protocol, configuration, source hashes,
  assignment audit, run-completion status, nine original latent reports,
  nine physical reports, and nine fitted ridge-decoder parameter files.
  summary_reference.json contains the numbers used in the manuscript.
  summarize.py runs on the included reports without original-server access.
  prepare.py, evaluate_physical.py, and run.py retain original paths and are
  archived experimental code, not portable launchers.

original_code/
  Original world-model trainer, latent evaluator, simulator replay generator,
  cache builder, source manifest builder, and source-split audit scripts.
  The trainer contains legacy CEReg/monitoring branches. The reported main
  experiment disables CEReg and the initial/final monitoring evaluations;
  consult config.json and training_metadata rather than header examples.

lewm_snapshot/
  jepa.py, module.py, original README, original download script, and MIT
  license from the recorded LeWorldModel checkout. This is a partial snapshot,
  not the complete upstream repository or installed Python environment.

data/
  Four exact non-image replay-array exports, the strict source manifest,
  per-policy cache manifests, original HDF5 attributes/inventory/hashes,
  and the installed-package version inventory from the experiment machine.

training_metadata/
  Saved training arguments, training curves, and other metadata for the
  nine principal checkpoints. World-model weights are excluded.

historical_reports/
  Existing factor-ablation, sampling, scaling, and control reports and audits.
  These are kept separate from the principal strict-split experiment.

manuscript_assets/
  Report inputs, figure/table generation scripts, and generated table sources.

tools/reproduce_and_verify.py
  Portable postprocessing/verification entry point added for this supplement.

5. INTERPRETATION
The main policies have three training seeds each: S=0,1,3; P/U=0,1,2.
The held-out MD condition combines familiar mass and delay values. Raw latent
rollout MSE favors paired collection, but this does not establish general
physical-position or control gains. No closed-loop control result is reported
for the nine principal checkpoints. The older scaling/control dataset has
32 source episodes overlapping between train and test; those reports are
development evidence only. The state/unpaired pool has 192 records but
191 unique episode/start keys. Pairing changes source reuse and is not an
isolated intervention on a pair-specific training loss.

6. PROVENANCE AND THIRD-PARTY MATERIAL
The original experiment scripts are preserved, including historical absolute
paths; no credentials are included. source_sha256.json records their hashes
at experiment launch. MANIFEST_SHA256.json covers immutable inputs; generated
summary/figure/table outputs and verification_report.json are excluded so
the documented reproduction commands can be rerun.
LeWorldModel components retain their original license and attribution in
lewm_snapshot/LICENSE. This archive does not grant a new blanket license
over third-party datasets or software. Cite the manuscript, LeWorldModel,
OGBench, and MuJoCo when using the corresponding material.
AI assistance in drafting and additional evaluation/postprocessing scripts
is disclosed in the manuscript acknowledgment. The portable supplement
verifier and packaging documentation were also prepared with OpenAI Codex.
