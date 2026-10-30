"""Fetch the recorded upstream source versions without building or running them."""
import argparse
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]

def main():
    sources=[r for r in json.loads((ROOT/'provenance/software.json').read_text()) if 'revision' in r]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods',nargs='+',choices=[r['name'] for r in sources],default=[r['name'] for r in sources])
    args=parser.parse_args()
    for row in sources:
        if row['name'] not in args.methods:continue
        destination=ROOT/row['path']
        if destination.exists():
            head=subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'],text=True).strip()
            if head!=row['revision']:raise ValueError(f'{destination} has a different revision; left unchanged')
            continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--no-checkout',row['repository'],str(destination)],check=True)
        subprocess.run(['git','-C',str(destination),'checkout','--detach',row['revision']],check=True)

if __name__=='__main__':main()
