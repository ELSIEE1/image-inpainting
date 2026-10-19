"""Portable launcher for the isolated, locally modified EdgeConnect code."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gan.yml')
    parser.add_argument('--checkpoints', type=Path, default=ROOT/'checkpoints/edge_connect')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/InpaintCOCO_export')
    parser.add_argument('--list-file', type=Path, default=ROOT/'splits/test_100.txt')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/gan_inpaintcoco')
    parser.add_argument('--mode', choices=['test', 'train', 'eval'], default='test')
    parser.add_argument('--dry-run', action='store_true', help='Validate paths/split without importing ML packages or loading weights.')
    args = parser.parse_args()
    for key in ('config', 'checkpoints', 'data_root', 'list_file', 'output_dir'):
        setattr(args, key, getattr(args, key).resolve())
    if not args.config.is_file():
        parser.error(f'Missing configuration: {args.config}')
    rows = [line.split() for line in args.list_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or any(len(row) != 3 for row in rows):
        parser.error('Split must contain input, mask, target columns.')
    print(f'Mode: {args.mode}; samples: {len(rows)}; config: {args.config}')
    if args.dry_run:
        missing = sum(not (args.data_root / value).is_file() for row in rows for value in row)
        print(f'Missing split files: {missing}; weights present: {args.checkpoints.is_dir()}')
        print('Dry run only; no model imports, downloads or output writes.')
        return
    command = [sys.executable, str(ROOT/'scripts/edgeconnect_worker.py')]
    for name in ('config', 'checkpoints', 'data_root', 'list_file', 'output_dir', 'mode'):
        command.extend(['--'+name.replace('_','-'), str(getattr(args,name))])
    subprocess.run(command, check=True, cwd=ROOT)

if __name__ == '__main__':
    main()
