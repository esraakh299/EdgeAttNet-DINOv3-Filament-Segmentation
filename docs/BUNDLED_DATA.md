# Bundled training metadata and masks

- `data/splits/physical_images_split.parquet` and `annotation_sets_split.parquet`: authoritative train/validation roster (566/141) and annotation sets.
- `data/splits/physical_images.parquet`, `annotation_sets.parquet`: original unfiltered metadata from supplied `seg_data.zip`.
- `data/assets/training_masks.zip`: exact supplied `masks.zip` with `semantic/` and `spine/`, each holding 1,154 PNG files.
- `data/ASSET_SHA256.json`: SHA256 integrity manifest.

The files were packaged without regenerating or modifying the contents of the uploaded archives. The verification script checks the parquet references against both mask namespaces when run in the Kaggle environment.

Run `python scripts/prepare_training_assets.py` to validate and extract masks to a writable cache. This is called by the retraining launcher; pretrained inference and OOF-fold inference do not need the masks.

Only the official competition data remains external for images and COCO annotations. The pre-trained model, trained CatBoost filter, four OOF models, and pretrained encoder remain separately distributed artifacts.
