"""Resolve and strictly load the frozen submission artifacts.

Only user-provided trusted .joblib files should be loaded: joblib uses pickle.
The committee path requires no Hugging Face authentication or manual download
when MODEL_DIR points to a mounted Kaggle dataset containing the three artifacts.
"""
from pathlib import Path
from collections.abc import Mapping
import hashlib
import json
import os

ORIGINAL_TRAINING_SNAPSHOT_SHA256 = 'f9439e623661bbb06158051d4f907d7f1cbfd5fc6d3082fd6aaae0881bc73970'
DEFAULT_HF_REPO = 'youssefElghanam191/edgeattnet-dinov3-convnext-base'
DEFAULT_HF_REVISION = 'f432bda8b3545e7f3736bdceecdb8698c406ff8c'

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def resolve_artifacts(model_dir=None, hf_repo=DEFAULT_HF_REPO, allow_download=True, revision=None):
    """Choose local/mounted files first; otherwise download publicly accessible files.

    Public downloads require no credentials. Do not prompt for tokens or silently
    fall back to another model if an artifact is missing.
    """
    if revision is None and hf_repo == DEFAULT_HF_REPO:
        revision = DEFAULT_HF_REVISION
    base=Path(model_dir).expanduser() if model_dir else Path(os.environ.get('EDGEATTNET_MODEL_DIR','models'))
    names={
        'weights': ('edgeattnet_dinov3_base_best.pth','dice_best_ema.pth'),
        'regressor': ('final_component_regressor_f.joblib',),
        'config': ('locked_filter_f.json',),
    }
    local={}
    for key,opts in names.items():
        matches=[base/x for x in opts if (base/x).is_file()]
        if matches:local[key]=matches[0]
    if len(local)==3:return local
    if not allow_download:
        raise FileNotFoundError(f'Incomplete model directory: {base}; missing {set(names)-set(local)}. '
                                'Attach the pretrained artifact dataset or make files publicly downloadable.')
    try:
        from huggingface_hub import hf_hub_download
        for key,opts in names.items():
            if key not in local:
                # The published upstream has the older checkpoint wrapper filename.
                remote='dice_best_ema.pth' if key=='weights' else opts[0]
                local[key]=Path(hf_hub_download(repo_id=hf_repo,filename=remote,token=False,revision=revision))
    except Exception as e:
        raise RuntimeError('Automatic artifact download failed. Provide the three files as a public '
                           'Kaggle dataset or unrestricted HF repository, or set MODEL_DIR to a local directory. '
                           'The demo never asks the committee to authenticate.') from e
    return local

def load_model(path, device):
    import torch
    from src.architecture import build_model
    blob=torch.load(path,map_location='cpu',weights_only=True)
    # Supports both a model-only state_dict and our legacy EMA snapshot wrapper.
    if isinstance(blob,Mapping) and 'model' in blob and isinstance(blob['model'],Mapping):
        if blob.get('variant','dino_convnext_base')!='dino_convnext_base':
            raise ValueError(f'Incorrect variant: {blob.get("variant")}')
        weights=blob['model']
    else:weights=blob
    if not isinstance(weights,Mapping) or not weights or not all(torch.is_tensor(t) for t in weights.values()):
        raise TypeError('Expected a state_dict or a checkpoint containing the model state_dict')
    net=build_model()
    net.load_state_dict(weights,strict=True)
    del blob,weights
    return net.to(device).eval().requires_grad_(False)

def load_filter(path,config):
    import joblib
    from src.filtering import ALL_FEATURES
    locked=json.loads(Path(config).read_text())
    ch=locked.get('choice')
    required={'algorithm','min_area','cutoff','low','seed'}
    if not isinstance(ch,dict) or not required<=set(ch):
        raise ValueError(f'Filter config must contain choice keys: {sorted(required)}')
    if locked.get('features') is not None and list(locked['features'])!=list(ALL_FEATURES):
        raise ValueError('Feature order differs from the saved filter configuration')
    # Trusted source only: joblib.load executes pickle data.
    return joblib.load(path),ch
