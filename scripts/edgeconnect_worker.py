"""Internal subprocess boundary prevents EdgeConnect's `src` from colliding with ours."""
import argparse
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoints','data-root','list-file','output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--mode', choices=['test','train','eval'], required=True)
    args = parser.parse_args()
    names = ['EdgeModel_gen.pth', 'InpaintingModel_gen.pth']
    if args.mode == 'test':
        for name in names:
            if not (args.checkpoints/name).is_file():
                parser.error(f'Missing checkpoint: {args.checkpoints/name}. See docs/pretrained_models.md; provenance requires confirmation.')
    sys.path.insert(0, str(ROOT/'third_party/edge_connect'))
    from src.config import Config
    config = Config(str(args.config))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    import cv2
    import numpy as np
    import torch
    from src.edge_connect import EdgeConnect
    config.PATH = str(args.checkpoints)
    config.MODE = {'train':1,'test':2,'eval':3}[args.mode]
    config.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(0)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    # Paths in the repository configuration are rooted at the repository.
    for field in ('TRAIN_FLIST','VAL_FLIST','TRAIN_EDGE_FLIST','VAL_EDGE_FLIST','TRAIN_MASK_FLIST','VAL_MASK_FLIST','TEST_EDGE_FLIST'):
        value = getattr(config, field)
        if value:
            setattr(config, field, str((ROOT/value).resolve()))
    if args.mode == 'test':
        rows = [line.split() for line in args.list_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        for row in rows:
            for value in row[:2]:
                if not (args.data_root/value).is_file():
                    parser.error(f'Missing dataset file: {args.data_root/value}')
        # Lists are passed directly; never commit machine-specific absolute flists.
        config.TEST_FLIST = [str(args.data_root/row[0]) for row in rows]
        config.TEST_MASK_FLIST = [str(args.data_root/row[1]) for row in rows]
    elif not config.TRAIN_FLIST or not config.VAL_FLIST:
        parser.error('Training/evaluation requires a separate config with TRAIN_FLIST and VAL_FLIST. The supplied config is test-only.')
    config.RESULTS = str(args.output_dir)
    model = EdgeConnect(config)
    model.load()
    {'test':model.test, 'train':model.train, 'eval':model.eval}[args.mode]()

if __name__ == '__main__':
    main()
