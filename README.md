# State and Physics Coverage for Visual World Models

Supplementary code and numerical data for **State and Physics Coverage for Visual World Models: A Fixed-Budget Empirical Study**.

The uploaded manuscript and original supplementary archive are included. The archive's 140 files are also extracted and organized below into code, data/results, and documentation. Their contents are unchanged; `SUPPLEMENTARY_FILE_MAP.json` records every original path, repository path, file size, and SHA-256 hash.

## Paper and downloads

- [Manuscript (PDF)](paper/ieee_access_2.pdf)
- [Original supplementary code and data (ZIP)](archive/IEEE_Access_Supplementary_Code_Data.zip)
- [Original supplementary README](docs/README.txt)
- [Data dictionary](docs/DATA_DICTIONARY.txt)
- [Requirements for retraining and visual re-evaluation](docs/FULL_EXPERIMENT_REQUIREMENTS.txt)

## Repository layout

| Folder | Contents |
| --- | --- |
| `code/original_code/` | Original training, latent evaluation, simulation, cache-building, and source-audit scripts. |
| `code/revision_minimal_20261030/` | Added evaluation and aggregation scripts, configuration, frozen protocol, environment record, and source hashes. |
| `code/tools/` | Portable numerical reproduction and verification script. |
| `code/manuscript_assets/scripts/` | Manuscript table and figure generation scripts. |
| `code/lewm_snapshot/` | Partial LeWorldModel source snapshot with its original README and MIT license. |
| `code/requirements-*.txt` | Numerical analysis and figure-generation dependencies. |
| `data/` | Four non-image replay-array exports, source and cache manifests, replay inventory, and package inventory. |
| `data/revision_minimal_20261030/` | Nine-checkpoint latent and physical reports, fitted ridge decoders, assignment audit, summaries, and completion records. |
| `data/training_metadata/` | Training arguments, curves, and metadata for the principal checkpoints. |
| `data/historical_reports/` | Earlier ablation, sampling, scaling, control, and audit reports. |
| `data/manuscript_assets/` | Archived report inputs, generated figures, and table sources. |
| `docs/` | Original supplementary documentation, integrity manifest, and archived verification report. |
| `paper/` | Uploaded manuscript PDF. |
| `archive/` | Unmodified original supplementary ZIP. |

## Reproduce the numerical results

The extracted `code/` and `data/` folders make the materials easy to browse separately. The original scripts use the archive's relative paths. **Run the following commands from the repository root to restore that original layout in a separate working folder before running them.** Python 3.10 or later is required.

```bash
python -m zipfile -e archive/IEEE_Access_Supplementary_Code_Data.zip reproduction
cd reproduction
python -m pip install -r requirements-analysis.txt
python tools/reproduce_and_verify.py
```

The portable verifier reaggregates the nine models over four test conditions with the original 10,000-replicate bootstrap, compares the results with the manuscript reference, checks non-image replay data, and writes `verification_report.json`. This is CPU-only numerical postprocessing; it does not train models or run visual inference.

To rebuild manuscript tables and figures, run these commands inside the same `reproduction/` folder:

```bash
python -m pip install -r requirements-figures.txt
python manuscript_assets/scripts/build_tables_figures.py
```

The original `MANIFEST_SHA256.json` is preserved in `docs/` and inside the ZIP. Its paths refer to the original archive layout. For the separated repository layout, consult [SUPPLEMENTARY_FILE_MAP.json](SUPPLEMENTARY_FILE_MAP.json).

## Scope and limitations

This supplement supports numerical reaggregation, manuscript table/figure reconstruction, and source/replay auditing. It includes original experiment scripts for inspection.

The source expert HDF5, full rendered replay HDF5 files with RGB images, world-model checkpoint weights, and intermediate latent trajectories are **not included**. The four full rendered replay files occupy approximately 550 MB in total. The `.npz` replay exports contain states and actions without images; the fitted ridge-decoder parameters are not world-model weights. See the [data dictionary](docs/DATA_DICTIONARY.txt) and [full experiment requirements](docs/FULL_EXPERIMENT_REQUIREMENTS.txt) for details.

Archived training and visual evaluation launchers retain historical server paths and require additional inputs and environment configuration. The partial LeWorldModel snapshot is not a complete upstream checkout. The original supplementary documentation does not claim public availability of the omitted artifacts.

The older scaling/control reports are development evidence and have source-episode overlap between training and test data. No closed-loop control result is reported for the nine principal checkpoints. Consult the manuscript and frozen protocol when interpreting the results.

## Attribution and contact

Authors: Yanfei Liu, Liangzhong Lin, Zhiying Cao, Chengliang Li, and Weichang Li.

Corresponding author: Weichang Li — aumax@mail.scut.edu.cn.

The LeWorldModel components retain their original [MIT license](code/lewm_snapshot/LICENSE). No new blanket license is granted over the manuscript, datasets, or other third-party software. Cite the manuscript, LeWorldModel, OGBench, and MuJoCo when using the corresponding material. Original provenance and AI-assistance disclosures are retained in the manuscript and supplementary documentation.
