# Method

## Data and preprocessing

The network receives a 2048 × 2048 H-alpha image converted to grayscale and
scaled to [0, 1]. The submitted preprocessing is implemented in
[`tta8_probability`](../src/inference.py). Multiple annotation records for one
physical image stay together in the bundled 566-training / 141-validation split.
Only the competition-provided annotations are used by the reconstruction code.

## Segmentation network

[`architecture.py`](../src/architecture.py) implements a DINOv3 ConvNeXt-Base
encoder and a multiscale EdgeAttNet decoder with two edge-guided multi-head
self-attention blocks. The selected historical checkpoint is the epoch-50 EMA
state. The full training recipe uses 1024-pixel crops, balanced augmentation,
AdamW, mixed precision, EMA 0.99 and a 55-epoch Dice stage; checkpoint selection
can choose an earlier epoch. See [training](TRAINING.md).

## From probabilities to instances

1. Evaluate all eight D4 rotations/reflections, undo each transform and average
   sigmoid probabilities at native resolution.
2. Quantize the average to FP16 and convert back to FP32 for postprocessing,
   preserving the original probability-cache precision.
3. Extract hysteresis components with support 0.15 and seed 0.90.
4. Compute 25 shape/probability features and six photometric/position features
   in the saved order; see [`filtering.py`](../src/filtering.py).
5. Keep components of at least 150 pixels whose CatBoost score is at least 0.25.
6. Encode each selected mask in Fortran-order COCO RLE and write the official
   `filament_id,segmentation_rle` CSV.

## Learned selector

Four training-only fold models generate out-of-fold maps. These models are not
an inference ensemble. The final CatBoost regressor uses depth 5 and 550 trees.
The published configuration describes five-fold date-block CV on the 566 OOF
images for selector selection; this is distinct from the four segmentation
folds. Historical model selection can introduce optimism; no tuning occurs in
the final inference notebook.

The original fold manifests are bundled under `data/oof/`. Companion metadata
is generated automatically after checking each published fold weight against
its pinned SHA-256. No extra JSON uploads are needed. See [OOF models](OOF_MODELS.md).

## Evaluation and interpretation

See [the metric protocol](EVALUATION.md). The notebook renders the first sorted
test image as a reproducible visual example, with probability and instance
outlines; it is not a best-case selection. This preview supports morphology
inspection but does not replace the organizers' broader quantitative and
qualitative assessment. The historical PQ 0.4751 is a reported result, not a
fresh measurement from this cleanup.

## References

- [Competition and evaluation requirements](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements)
- [Organizer self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
- [MAGFiLO dataset paper](https://doi.org/10.1038/s41597-024-03876-y)
- [DINOv3 source and pretrained models](https://github.com/facebookresearch/dinov3)
- [Published model artifacts](https://huggingface.co/youssefElghanam191/edgeattnet-dinov3-convnext-base)

Third-party model and dataset terms remain applicable. This cleanup does not
assign a new license to the team's source code.
