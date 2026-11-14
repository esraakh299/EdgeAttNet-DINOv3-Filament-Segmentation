"""Artifact resolution must be offline-safe and reproducible."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from src.artifacts import resolve_artifacts, DEFAULT_HF_REVISION

NAMES = ['dice_best_ema.pth', 'final_component_regressor_f.joblib', 'locked_filter_f.json']

def test_complete_local_bundle_never_downloads(tmp_path, monkeypatch):
    for name in NAMES:
        (tmp_path/name).write_bytes(b'fixture')
    def forbidden(**kwargs):
        raise AssertionError('Local resolution attempted a download')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=forbidden))
    result = resolve_artifacts(tmp_path)
    assert {p.name for p in result.values()} == set(NAMES)

def test_incomplete_offline_bundle_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match='missing'):
        resolve_artifacts(tmp_path, allow_download=False)

def test_downloads_use_anonymous_pinned_revision(tmp_path, monkeypatch):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path/kwargs['filename'])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    result = resolve_artifacts(tmp_path)
    assert len(result) == len(calls) == 3
    assert all(c['token'] is False and c['revision'] == DEFAULT_HF_REVISION for c in calls)

def test_custom_repository_does_not_inherit_unrelated_commit(tmp_path, monkeypatch):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path/kwargs['filename'])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    resolve_artifacts(tmp_path, hf_repo='team/custom')
    assert all(c['revision'] is None for c in calls)
