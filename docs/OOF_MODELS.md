# OOF reconstruction

The four fold models create held-out probability maps for training the CatBoost
selector. They are not an ensemble used by the final inference notebook.

## Run with the published models

1. Install `requirements.txt` and attach the official competition data.
2. Run `notebooks/04_OOF_Folds_01.ipynb` and
   `notebooks/05_OOF_Folds_23.ipynb` with `TRAIN_OOF_FROM_SCRATCH = False`.
3. Use the resulting pair-output directories in `notebooks/02_Filter_Training.ipynb`
   and set `TRAIN_FILTER_FROM_SCRATCH = True` to refit the selector.

The published files are named `fold_N_model.pth` but contain original EMA
snapshot wrappers, including their fold metadata. The loader recognizes this
format and the inference worker verifies the embedded training/held-out rosters
before loading the model tensors. All four wrappers were downloaded and inspected.

The loader downloads only the requested fold weights anonymously from the
revision pinned in `data/oof/release.json`. It checks their published SHA-256
values and **generates each companion JSON automatically** from the original
manifests bundled in `data/oof/`. No separate JSON uploads, original checkpoint
wrappers, encoder downloads or fold retraining are needed for this path.

The two supplied pair manifests are byte-identical and both describe all four
folds. Their original filenames are retained. Their signatures, checksums,
566-image held-out coverage and 141-image validation separation are checked
before sidecars are generated. The runtime also reconstructs the original
fold manifest from the actual competition data and compares the metadata
against that manifest before inference.

The binding between the named published fold weights and training manifests is
the team's release record. Hash checks establish file identity; tensor values
do not independently establish a model's training history.

## Offline or mounted weights

Point `OOF_MODELS_ROOT` at a directory containing `fold_0_model.pth` through
`fold_3_model.pth`. Missing sidecars for the pinned release are generated in the
writable output directory, so read-only Kaggle inputs work. Disable downloads
when all required weights are mounted. Unknown or modified weights are rejected;
custom model releases must supply their own valid provenance sidecars.

For an optional standalone preparation step:

```bash
python scripts/prepare_oof_metadata.py --models /path/to/oof --out outputs/oof_metadata
```

This only hashes the files and writes JSON. It uses no GPU and does not export
or change model tensors. The notebooks perform this step automatically.

## Probability maps and filter fitting

All 566 held-out maps can be regenerated from the fold models; no map upload is
required. Their generation uses native-resolution TTA8 on a CUDA GPU and takes
substantially longer than writing metadata. Original cached maps may be reused
only when their model hashes and per-image sidecars match the current artifacts.

After both pairs finish:

```bash
python scripts/audit_oof_coverage.py \
  --pair-01 /path/to/DINO_OOF_Folds01 \
  --pair-23 /path/to/DINO_OOF_Folds23 \
  --out outputs/oof_verified
python scripts/fit_filter_oof.py \
  --dataset /path/to/MAGFiLO_1.0_Kaggle_2026 \
  --splits data/splits --oof outputs/oof_verified --out outputs/learned_filter
```

The audit rejects duplicate, incomplete and inconsistent held-out maps. Filter
refitting fixes the selected CatBoost hyperparameters; it does not repeat the
historical search and is not guaranteed to produce identical model bytes.

## Optional fresh training

Set `TRAIN_OOF_FROM_SCRATCH = True` and provide the licensed DINOv3 ConvNeXt-Base
encoder initialization. The original recipe uses four temporal GroupKFold
splits, a two-day embargo, seed 42 and fixed epoch-55 EMA checkpoints. The paired
training workflow expects two GPUs; pretrained inference supports one GPU.
Preserve the entire output to resume interrupted sessions.

For a new model release, `scripts/export_oof_models.py` exports original EMA
wrappers and their provenance. `scripts/upload_oof_to_hf.py` remains available
for explicit publication. Automatic sidecar reconstruction applies only to the
checksum-pinned release bundled here, not to arbitrary newly trained models.
