"""Non-GPU contract checks for the two original paired four-fold OOF runners."""
from pathlib import Path
import hashlib,json
import nbformat
import numpy as np
import torch
import pytest
from src.oof_helpers import (fold_signature, map_signature, sha256_file,
                             discover_fold_weights, install_precomputed_maps,verify_pair_completion)

ROOT=Path(__file__).resolve().parents[1]


def fake_manifest():
    return dict(signature='original-manifest',validation_names=['never-train.jpeg'],
                image_hashes={'heldout.jpeg':'sha-of-image'},
                hyperparameters={'epochs':55},
                folds=[dict(training_names=['different.jpeg'],heldout_names=['heldout.jpeg'])]+
                      [dict(training_names=[],heldout_names=[]) for _ in range(3)])


def test_oof_pair_notebooks_offer_disabled_training_by_default():
    for pair in ['01','23']:
        p=ROOT/'notebooks'/f'{"04" if pair=="01" else "05"}_OOF_Folds_{pair}.ipynb'
        n=nbformat.read(p,as_version=4);nbformat.validate(n)
        code='\n'.join(c.source for c in n.cells if c.cell_type=='code')
        assert 'TRAIN_OOF_FROM_SCRATCH = False' in code
        assert "'--folds',FOLD_PAIR" in code
        assert 'REUSE_PRECOMPUTED_OOF_MAPS = True' in code
        for c in n.cells:
            if c.cell_type=='code':compile(c.source,str(p),'exec')


def test_model_only_provenance_requires_exact_folds(tmp_path):
    m=fake_manifest()
    out=tmp_path/'fold_0_model.pth';state={f'layer.{i}':torch.zeros(1) for i in range(301)}
    torch.save(state,out)
    d=dict(model_sha256=sha256_file(out),manifest_signature=m['signature'],
           signature=fold_signature(m,0),fold=0,training_names=['different.jpeg'],
           heldout_names=['heldout.jpeg'],epoch=55,selection='fixed_final_epoch_EMA',
           original_validation_used=False,heldout_labels_used=False)
    out.with_suffix('.json').write_text(json.dumps(d))
    result=discover_fold_weights(tmp_path,m,[0])
    assert result[0]['format']=='model_only'
    d['heldout_names']=['never-train.jpeg']
    out.with_suffix('.json').write_text(json.dumps(d))
    with pytest.raises(ValueError):discover_fold_weights(tmp_path,m,[0])


def test_reuse_cached_oof_checks_map_sha_and_original_checkpoint(tmp_path):
    m=fake_manifest();src=tmp_path/'source';dst=tmp_path/'output'
    maps=src/'prob_oof'/'fold_0';maps.mkdir(parents=True)
    weights=src/'fold_0'/'fixed_final_ema.pth';weights.parent.mkdir(parents=True)
    weights.write_bytes(b'unit-test original fold model')
    rec={'format':'snapshot','sha256':sha256_file(weights),'checkpoint':weights}
    path=maps/'heldout.jpeg.npy';np.save(path,np.array([[.2]],dtype=np.float16))
    meta=dict(signature=map_signature(m,0,rec['sha256']),manifest_signature=m['signature'],
              checkpoint_sha256=rec['sha256'],fold=0,file_name='heldout.jpeg',
              image_sha256='sha-of-image',image_excluded_from_model_training=True,
              map_sha256=sha256_file(path),mode='full',tta8=True)
    (maps/'heldout.jpeg.json').write_text(json.dumps(meta))
    assert install_precomputed_maps(src,dst,m,0,rec)
    assert verify_pair_completion(dst,m,[0])[0]['complete']
    meta['image_excluded_from_model_training']=False
    (maps/'heldout.jpeg.json').write_text(json.dumps(meta))
    assert not install_precomputed_maps(src,tmp_path/'other',m,0,rec)


def test_oof_worker_is_archived_from_original_recipe():
    s=(ROOT/'src'/'oof_training_worker.py').read_text()
    assert 'def train_fold(' in s and 'def infer_fold(' in s
    assert 'fixed_final_ema.pth' in s and "net.load_state_dict(state,strict=True)" in s
    assert "['training_names']" in s and "['heldout_names']" in s
