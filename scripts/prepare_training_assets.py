"""Prepare canonical training masks from the bundled archive.

No Kaggle input other than the official competition dataset is necessary.
Splits remain versioned under data/splits; mask archive is checked by SHA256.
"""
import argparse, hashlib, json, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MASK_ARCHIVE=ROOT/'data'/'assets'/'training_masks.zip'
SPLITS=ROOT/'data'/'splits'
CHECKS=ROOT/'data'/'ASSET_SHA256.json'

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def prepare(output=None):
 import pandas as pd
 if output is None:
  output=Path('/kaggle/working/edgeattnet_training_assets') if Path('/kaggle/working').is_dir() else ROOT/'outputs'/'training_assets'
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 expected=json.loads(CHECKS.read_text())
 for rel,digest in expected.items():
  file=ROOT/rel
  if not file.is_file() or sha256(file)!=digest:
   raise RuntimeError(f'Bundled asset missing or modified: {file}')
 physical=pd.read_parquet(SPLITS/'physical_images_split.parquet')
 annotations=pd.read_parquet(SPLITS/'annotation_sets_split.parquet')
 if (physical['split']=='train').sum()!=566 or (physical['split']=='val').sum()!=141:
  raise ValueError('Curated split mismatch; expected 566 train and 141 validation physical images')
 needed=set(annotations['mask_file'].dropna().astype(str))
 if not needed:raise RuntimeError('No annotation mask filenames in bundled split table')
 with zipfile.ZipFile(MASK_ARCHIVE) as z:
  known=set(z.namelist())
  if len(known)!=len(z.namelist()):raise RuntimeError('Duplicate mask filenames in archive')
  missing=sorted(f'{sub}/{name}' for name in needed for sub in ('semantic','spine') if f'{sub}/{name}' not in known)
  if missing:raise FileNotFoundError(f'Mask archive missing {len(missing)} referenced masks: {missing[:5]}')
  for sub in ('semantic','spine'):
   (output/sub).mkdir(exist_ok=True)
   for name in sorted(needed):
    if Path(name).name!=name or name.startswith('.') or '/' in name or '\\' in name:
     raise ValueError(f'Unsafe mask filename in metadata: {name!r}')
    target=output/sub/name
    if not target.exists() or target.stat().st_size!=z.getinfo(f'{sub}/{name}').file_size:
     target.write_bytes(z.read(f'{sub}/{name}'))
 for sub in ('semantic','spine'):
  assert all((output/sub/name).is_file() for name in needed)
 print(f'Verified {len(needed)} paired annotation masks in {output}; split 566/141')
 return output

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=None)
 args=parser.parse_args();prepare(args.output)
