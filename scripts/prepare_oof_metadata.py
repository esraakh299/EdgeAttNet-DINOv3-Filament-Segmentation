"""Optionally prepare OOF sidecars offline; the notebook also does this automatically."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oof_provenance import generate_sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True, help='Directory containing fold_N_model.pth')
    parser.add_argument('--out', type=Path, help='Writable metadata directory; defaults to model directory')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args()
    for fold in args.folds:
        output = (args.out or args.models) / f'fold_{fold}_model.json'
        print(generate_sidecar(args.models/f'fold_{fold}_model.pth', fold, output))


if __name__ == '__main__':
    main()
