"""Anonymous, revision-pinned OOF downloads with automatic local sidecars."""
from pathlib import Path
import shutil
from src.oof_provenance import generate_sidecar, release_info

DEFAULT_HF_REPO = 'youssefElghanam191/edgeattnet-dinov3-convnext-base'


def acquire_oof_weights(local_root, folds, repo_id=DEFAULT_HF_REPO, enabled=True, revision=None):
    """Reuse local files or fetch weights, then generate published-release metadata.

    The default release needs no hosted fold_N_model.json files. Custom model
    repositories still require their own sidecars. Local files work offline.
    """
    release = release_info()
    if revision is None and repo_id == DEFAULT_HF_REPO:
        revision = release['revision']
    bundled = repo_id == release['repo_id'] and revision == release['revision']
    downloaded = Path(local_root) / 'oof'
    downloaded.mkdir(parents=True, exist_ok=True)
    pairs = {}
    for fold in folds:
        if fold not in range(4):
            raise ValueError(f'Invalid fold: {fold}')
        name = f'fold_{fold}_model'
        weights = downloaded / (name + '.pth')
        sidecar = downloaded / (name + '.json')
        if weights.is_file() and sidecar.is_file():
            pairs[fold] = (weights, sidecar)  # Strict validation occurs in discovery.
            continue
        if not weights.is_file() and not enabled:
            continue
        if not weights.is_file():
            _download(repo_id, f'oof/{name}.pth', revision, weights)
        if bundled:
            generate_sidecar(weights, fold, sidecar)
        elif not sidecar.is_file():
            if not enabled:
                continue
            _download(repo_id, f'oof/{name}.json', revision, sidecar)
        pairs[fold] = (weights, sidecar)
    return pairs


def _download(repo_id, filename, revision, destination):
    try:
        from huggingface_hub import hf_hub_download
        cached = Path(hf_hub_download(repo_id=repo_id, repo_type='model',
                                     filename=filename, token=False, revision=revision))
    except Exception as exc:
        raise RuntimeError(f'Cannot anonymously download {filename} from {repo_id}. '
                           'Check Internet access or attach the model files locally.') from exc
    temporary = destination.with_suffix(destination.suffix + '.part')
    try:
        shutil.copy2(cached, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
