"""Reconstruct sidecars for the published OOF weights from bundled manifests.

Only checksum-pinned release weights qualify. The association with the original
training manifest is supplied by the team, not inferred from tensor values.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile

BUNDLE = Path(__file__).resolve().parents[1] / 'data' / 'oof'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def release_info():
    return json.loads((BUNDLE / 'release.json').read_text())


def load_manifest(fold):
    if fold not in range(4):
        raise ValueError(f'Invalid fold: {fold}')
    release = release_info()
    entry = release['folds'][str(fold)]
    path = BUNDLE / entry['manifest']
    if sha256(path) != entry['manifest_sha256']:
        raise ValueError('Bundled OOF manifest checksum mismatch')
    manifest = json.loads(path.read_text())
    body = {k: v for k, v in manifest.items() if k != 'signature'}
    signature = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if signature != manifest['signature']:
        raise ValueError('Invalid original OOF manifest signature')
    train = set(manifest['training_names'])
    val = set(manifest['validation_names'])
    heldout = [name for f in manifest['folds'] for name in f['heldout_names']]
    if (len(train) != 566 or len(val) != 141 or train & val
            or len(heldout) != len(set(heldout)) or set(heldout) != train):
        raise ValueError('Invalid original OOF coverage or validation separation')
    for index, item in enumerate(manifest['folds']):
        fit, hold = set(item['training_names']), set(item['heldout_names'])
        if item['fold'] != index or not fit <= train or not hold <= train or fit & hold:
            raise ValueError('Invalid original fold roster')
        if (item['initialization'] != 'DINOv3_ConvNeXtBase_LVD1689M'
                or item['fixed_epochs'] != 55 or manifest['hyperparameters']['epochs'] != 55
                or item['checkpoint_selection'] != 'fixed_final_epoch55_EMA; no heldout/validation labels inspected'):
            raise ValueError('Unexpected original fold training recipe')
    return manifest, entry, release


def generate_sidecar(weights, fold, output=None):
    """Hash weights, validate the original manifest and atomically write metadata.

    No GPU, tensor deserialization, model export or training is performed.
    Unknown/replaced weights fail rather than receiving fabricated provenance.
    """
    weights = Path(weights)
    manifest, entry, release = load_manifest(fold)
    digest = sha256(weights)
    if digest != entry['model_sha256']:
        raise ValueError(f'Fold {fold} weights do not match the published OOF release: {weights}')
    item = manifest['folds'][fold]
    payload = {
        'format': entry.get('format', 'pytorch_state_dict'),
        'fold': fold,
        'epoch': item['fixed_epochs'],
        'signature': hashlib.sha256((manifest['signature'] + f'/fold{fold}').encode()).hexdigest(),
        'manifest_signature': manifest['signature'],
        'training_names': item['training_names'],
        'heldout_names': item['heldout_names'],
        'selection': 'fixed_final_epoch_EMA',
        'initialization': item['initialization'],
        'original_validation_used': False,
        'heldout_labels_used': False,
        'model_sha256': digest,
        'provenance_source': release['provenance_source'],
        'release_revision': release['revision'],
        'original_manifest_sha256': entry['manifest_sha256'],
    }
    output = Path(output) if output is not None else weights.with_suffix('.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + '.', dir=output.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(payload, stream, indent=2)
            stream.write('\n')
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output
