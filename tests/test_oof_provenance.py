"""Automatic OOF metadata requires the original roster and exact release weights."""
import json
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from src import oof_provenance as provenance
from src.hf_oof import acquire_oof_weights
from src.oof_helpers import discover_fold_weights


@pytest.mark.parametrize('fold', range(4))
def test_original_manifests_cover_training_only(fold):
    manifest, entry, release = provenance.load_manifest(fold)
    assert len(manifest['training_names']) == 566
    assert len(manifest['validation_names']) == 141
    assert len(entry['model_sha256']) == 64


@pytest.fixture
def release_fixture(tmp_path, monkeypatch):
    bundle = tmp_path/'bundle'
    shutil.copytree(provenance.BUNDLE, bundle)
    model = tmp_path/'fold_0_model.pth'
    model.write_bytes(b'known fixture model')
    release = json.loads((bundle/'release.json').read_text())
    release['folds']['0']['model_sha256'] = provenance.sha256(model)
    (bundle/'release.json').write_text(json.dumps(release))
    monkeypatch.setattr(provenance, 'BUNDLE', bundle)
    return model, bundle


def test_generated_metadata_passes_existing_discovery(release_fixture, tmp_path):
    model, _ = release_fixture
    manifest, _, _ = provenance.load_manifest(0)
    # Separate writable directory models Kaggle's read-only input mounts.
    result = discover_fold_weights(model.parent, manifest, [0], metadata_root=tmp_path/'metadata')
    assert result[0]['metadata'].parent == tmp_path/'metadata'
    assert result[0]['format'] == 'snapshot'
    assert not model.with_suffix('.json').exists()
    assert json.loads(result[0]['metadata'].read_text())['initialization'] == 'DINOv3_ConvNeXtBase_LVD1689M'


def test_changed_weight_is_rejected(release_fixture):
    model, _ = release_fixture
    model.write_bytes(b'wrong fold or modified model')
    with pytest.raises(ValueError, match='do not match'):
        provenance.generate_sidecar(model, 0)
    assert not model.with_suffix('.json').exists()


def test_corrupt_manifest_is_rejected(release_fixture):
    model, bundle = release_fixture
    (bundle/'oof_training_manifest01.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        provenance.generate_sidecar(model, 0)


def test_offline_weights_get_sidecars_without_network(release_fixture, tmp_path, monkeypatch):
    model, _ = release_fixture
    destination = tmp_path/'local'/'oof'
    destination.mkdir(parents=True)
    shutil.copy2(model, destination/model.name)
    def forbidden(**kwargs):
        raise AssertionError('Offline run attempted network access')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=forbidden))
    result = acquire_oof_weights(destination.parent, [0], enabled=False)
    assert result[0][1].is_file()


def test_public_download_needs_only_weights(release_fixture, tmp_path, monkeypatch):
    model, _ = release_fixture
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(model)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    result = acquire_oof_weights(tmp_path/'download', [0])
    assert result[0][1].is_file()
    assert len(calls) == 1 and calls[0]['filename'] == 'oof/fold_0_model.pth'
    assert calls[0]['token'] is False
    assert calls[0]['revision'] == provenance.release_info()['revision']
