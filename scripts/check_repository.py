"""Dependency-light syntax, immutable-artifact, and publication-manifest checks."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='store_true', help='Print exact proposed first-commit file list.')
    args = parser.parse_args()
    tracked = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    paths = sorted({p for p in tracked if p and (ROOT/p).is_file()})
    failures=[]
    count=0
    for p in paths:
        if Path(p).parts[0] in {'.local', 'data', 'checkpoints', 'models', 'outputs', '.venv', 'venv', 'env'}:
            failures.append('Local-only directory in publication manifest: '+p)
        if p.endswith('.py'):
            ast.parse((ROOT/p).read_text(encoding='utf-8-sig'), filename=p)
            count+=1
        if (ROOT/p).stat().st_size > 10*1024*1024:
            failures.append('Unexpected publication file >10 MiB: '+p)
        if '.git' in Path(p).parts or Path(p).suffix in {'.pth','.pt','.ckpt','.safetensors','.pyc'}:
            failures.append('Excluded artifact in manifest: '+p)
        if p.startswith('results/') and not p.startswith('results/metrics/historical/'):
            failures.append('Generated output in publication manifest: '+p)
    records=json.loads((ROOT/'docs/artifact_checksums.json').read_text())
    for p, expected in records.items():
        actual=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
        if actual != expected: failures.append('Immutable artifact changed: '+p)
    lines=(ROOT/'splits/test_100.txt').read_text().splitlines()
    assert len(lines)==100 and len(set(lines))==100
    assert all(len(line.split())==3 for line in lines)
    print(f'Syntax: {count} Python files; immutable artifacts: {len(records)}; split: 100 unique records.')
    print(f'Proposed publication: {len(paths)} files, {sum((ROOT/p).stat().st_size for p in paths)/1048576:.2f} MiB.')
    if args.manifest:
        print('\n'.join(paths))
    if failures: raise SystemExit('\n'.join(failures))
    print('Repository checks passed.')

if __name__=='__main__':
    main()
