"""List or download the exact Hi-C inputs; downloading requires --download."""
import argparse
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',choices=['benchmark','nipbl','all'],default='benchmark')
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    for row in json.loads((ROOT/'provenance/hic_inputs.json').read_text()):
        if args.group!='all' and row['group']!=args.group:continue
        output=ROOT/row['path']
        print(f"{row['accession']}: {row.get('bytes', 'unknown')} bytes -> {row['path']}",flush=True)
        if not args.download:continue
        if output.exists():
            if row.get('bytes') and output.stat().st_size!=row['bytes']:
                raise ValueError(f'Existing input has unexpected size: {output}')
            continue
        output.parent.mkdir(parents=True,exist_ok=True)
        partial=output.with_name(output.name+'.partial')
        subprocess.run(['curl','--fail','--location','--retry','3','--continue-at','-','--output',str(partial),row['url']],check=True)
        if row.get('bytes') and partial.stat().st_size!=row['bytes']:
            raise ValueError(f'Unexpected download size: {partial}')
        partial.replace(output)

if __name__=='__main__':main()
