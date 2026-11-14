# Submission checklist

Official source checked 2026-11-12:
[competition instructions](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements).

| Requirement | Repository support / remaining action |
| --- | --- |
| Public Git repository | Publish this cleaned source; keep it public through the winner announcement |
| Packages and versions | Exact direct-dependency pins in `requirements.txt`; validate on the selected GPU runtime |
| Pipeline notebook | `final_inference.ipynb` covers inputs through final CSV |
| Complete reproducibility materials | Fixed splits and targets included; automatic inference artifact downloads configured |
| Pipeline description and readable code | `docs/METHOD.md`, focused entry point and modular source |
| Morphology inspection | Notebook exports an image/probability/instance preview |
| Four-page PDF in organizer template | Not supplied in the uploaded archive; prepare separately |
| Organizer submission form | Team must submit the final repository URL and report through the official form |

The rubric combines quantitative assessment (70%) with qualitative assessment
(30%). Qualitative review includes pipeline explanation, morphology and code
quality. Quantitative criteria include PQ and additional overlap/relationship
distributions; the bundled evaluator computes pooled PQ/SQ/RQ and counts. Use
the organizer self-evaluation notebook for their additional diagnostics.

## Required release acceptance run

- [ ] Install `requirements.txt` in a fresh CUDA environment and restart the kernel.
- [x] Run the default notebook against all 141 validation and 180 test images (15 November 2026: validation PQ 0.4751 reproduced exactly).
- [ ] Save `environment.txt`, `run_manifest.json`, validation CSVs and `preview.png`.
- [ ] Compare the generated CSV with the actual selected competition submission.
- [x] Confirm all model files download without authentication in a clean session.
- [ ] Replace proposed dependency pins with the verified runtime's versions if needed.
- [ ] Add the final report and complete the organizer form.

## Findings from this cleanup

- The inference model, filter and configuration are listed in the public model
  repository at revision `f432bda8b3545e7f3736bdceecdb8698c406ff8c`.
  The configuration was retrieved anonymously and matches 0.15 / 0.90 / 150 / 0.25.
  Both inference binaries returned HTTP 200 to anonymous HEAD requests. Full
  weight downloads and strict CPU loading have now passed; GPU inference remains
  the final acceptance check.
- OOF metadata is now generated automatically from the two original manifests.
  The four published weights are pinned by revision and SHA-256 in `data/oof/release.json`.
  No separate fold JSON uploads are required.
- A complete historical environment freeze and the selected CUDA-13 submission
  CSV were absent. Do not claim byte-for-byte parity until the acceptance run.
- The archived CUDA-12 CSV was removed: it was not the selected submission.

The competition states 15 November 2026 as the report/solution deadline and
30 November 2026 for winner announcements. Confirm the exact closing time on
Kaggle before submitting.

## Checks completed on this source package

- 22 CPU tests passed, including anonymous pinned-download and offline cases.
- All Python files compile; all five notebook schemas and code cells validate.
- All relative documentation links resolve.
- All five bundled asset SHA-256 values match the supplied archive.
- The 707-image / 1,154-annotation split loads and all 1,154 semantic/spine
  target pairs extract successfully.
- Architecture, data loading, filtering and inference source remain byte-identical
  to the upload. Changes concern packaging, artifact resolution, notebook paths,
  provenance recording and training launcher paths.

CPU checks ran on Python 3.12 with PyTorch 2.11.0+cpu in a test environment;
they do not validate the proposed CUDA environment or establish the reported PQ.

## OOF release verification — 2026-11-12

All four hosted fold weights were downloaded anonymously at revision
`19cc13b207a30795459b86d80cc36662fff289a0` and matched their published SHA-256.
They contain original epoch-55 EMA wrappers (despite their `fold_N_model.pth`
filenames). Their embedded fold numbers and training/held-out rosters match the
supplied original manifests. All four 431-tensor models load strictly on CPU.

The actual fold-building function reproduces the original manifest exactly
using bundled split tables and the recorded image hashes. This check does not
re-hash competition images, which are unavailable in the packaging environment.
The GPU runner still verifies the actual image hashes when it executes.

Generated sidecars work with the existing fold discovery path, including
read-only model mounts with a separate writable metadata directory. Tests
cover offline generation, anonymous pinned downloads, incorrect weights and
corrupted manifests. No further OOF JSON uploads are needed.

The final segmentation checkpoint also loads strictly on CPU. The published
CatBoost artifact loads and successfully scores a synthetic candidate through
the actual feature-extraction and instance-selection functions. These are
artifact compatibility checks, not a new validation-PQ measurement.
