"""Lightweight checks that don't need model weights or a GPU."""
import json
import nbformat
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_demo_has_training_toggle_and_outputs():
    nb=nbformat.read(ROOT/'final_inference.ipynb',as_version=4)
    code='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    assert 'TRAIN_FROM_SCRATCH = False' in code
    assert 'run_validation(' in code
    assert 'run_test(' in code
    assert 'run_manifest.json' in code
def test_model_only_converter_not_exporting_optimizer():
    s=(ROOT/'scripts'/'export_model.py').read_text()
    assert 'torch.save({k:v.cpu().contiguous()' in s
def test_no_credential_prompt():
    code=(ROOT/'src'/'artifacts.py').read_text()
    assert 'token=False' in code
