# Evaluation

The bundled split contains 566 training and 141 validation physical images.
All annotator views of each image stay on the same side of the split.

For every image–annotator pair, match instances at **IoU > 0.5**. Pool matched
IoU and TP/FP/FN across all pairs, then calculate:

- PQ = sum IoU / (TP + 0.5 FP + 0.5 FN)
- SQ = sum IoU / TP
- RQ = TP / (TP + 0.5 FP + 0.5 FN)

This is not the mean of per-image PQ. Empty-count conventions and implementation
are in [`filtering.py`](../src/filtering.py). The inference notebook applies
fixed artifact settings and performs no threshold search.

The reported validation PQ **0.4751** comes from the supplied repository, not a
fresh execution. The hosted configuration also contains selector-CV metrics;
those are not the 141-image validation score.

## Runtime and parity

The supplied README selected PyTorch 2.11.0 / CUDA 13.0 for final inference.
Older notebook text instead described CUDA 12.8, which reproduced one archived
probability map. Small numerical differences can change a borderline filter
decision. The cleaned documentation uses the selected CUDA 13.0 runtime, but a
full GPU run and comparison with the selected CSV are still required.

The notebook records artifact/split SHA-256 values, package versions, GPU,
CUDA, cuDNN and precision settings. Proposed package pins are not a recovered
historical environment. Preserve the generated evidence with the final release.

For the organizers' additional Dice/IoU and split/merge diagnostics, use their
[self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook).
