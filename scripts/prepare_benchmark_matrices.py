"""Recreate all 154 benchmark matrices from the published Hi-C files."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from recompute_published_tad_metrics import CONDITIONS, BASE
from run_tad_all_chromosomes import materialize, SOURCES
ROOT = Path(__file__).resolve().parents[1]
ACCESSIONS = {'HIC002':'GSM1551551','HIC074':'GSM1551623','HIC056':'GSM1551605'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', nargs='+', choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument('--chromosomes', nargs='+', type=int, default=list(range(1,23)))
    args = parser.parse_args()
    if any(not 1 <= n <= 22 for n in args.chromosomes):
        parser.error('Expected autosomes 1 through 22')
    for condition in args.conditions:
        prefix, resolution = CONDITIONS[condition]
        for chromosome in args.chromosomes:
            chrom = f'chr{chromosome}'
            output = BASE/condition/'matrices'/f'{prefix}_{resolution//1000}k_KR.{chrom}'
            expected = json.loads((ROOT/'artifacts/tad_no_separator_20261029/benchmark'/condition/f'{chrom}.json').read_text())['matrix_shape']
            output.parent.mkdir(parents=True, exist_ok=True)
            if resolution==5000:
                output=output.with_suffix(output.suffix+'.npy')
                if not output.exists():
                    accession=SOURCES[condition.split('_')[0]][0]
                    source=ROOT/f'data/raw/hic_neural/{accession}.mcool'
                    if not source.exists():raise FileNotFoundError(source)
                    materialize(source,chrom,output)
                shape=list(np.load(output,mmap_mode='r').shape)
            else:
                if not output.exists():
                    source=ROOT/f'data/raw/tad_original_hic/{ACCESSIONS[prefix]}_{prefix}.hic'
                    if not source.exists():raise FileNotFoundError(source)
                    temporary=output.with_name(output.name+'.juicer-tmp')
                    subprocess.run(['java','-Xmx4g','-jar',str(ROOT/'third_party/deepTAD/prepare_data/juicer_tools.jar'),
                                    'dump','observed','KR',str(source),chrom,chrom,'BP',str(resolution),str(temporary),'-d'],check=True)
                    subprocess.run([sys.executable,str(ROOT/'third_party/deepTAD/prepare_data/remove_nan.py'),str(temporary),str(output)],check=True)
                # The historical coarse inputs use the upstream text format.
                shape=list(np.loadtxt(output).shape)
            if shape!=expected:raise ValueError(f'{condition} {chrom}: {shape} != archived {expected}')
            print(condition,chrom,shape,flush=True)

if __name__=='__main__':main()
