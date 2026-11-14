# Inference artifacts

The default notebook retrieves these files anonymously from
[the model repository](https://huggingface.co/youssefElghanam191/edgeattnet-dinov3-convnext-base)
at revision `f432bda8b3545e7f3736bdceecdb8698c406ff8c`.

| File | Role |
| --- | --- |
| `dice_best_ema.pth` | Selected segmentation EMA snapshot; loaded with `weights_only=True` |
| `final_component_regressor_f.joblib` | Trained 31-feature CatBoost selector |
| `locked_filter_f.json` | Feature order and fixed decision thresholds |

For offline use, place all three here or set `MODEL_DIR`. A model-only export
named `edgeattnet_dinov3_base_best.pth` is also accepted locally and takes
precedence over the wrapped snapshot. All model keys are loaded strictly.
Only use trusted model/filter files; joblib is a pickle-based format.

These binaries are not embedded in the source ZIP. Anonymous API listing and
configuration retrieval and anonymous binary HEAD access were checked on 2026-11-12; full binary download and strict CPU model loading also passed. A complete
GPU inference run remains the release acceptance check. The manifest records the actual files' SHA-256 hashes.

Optional OOF reconstruction downloads the four fold weights at the revision
in `data/oof/release.json` and generates their companion JSON files automatically
from the bundled original manifests. No separate sidecar uploads are needed.
Instructions: [OOF models](../docs/OOF_MODELS.md).
