"""Evaluate new TAD outputs without overlapping timed caller executions."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import fcntl
import os
import time
import sys

from tad_result_state import ROOT, BASE, EXTRA, records, cached, fingerprint, read, write
sys.path.insert(0,str(ROOT/'src'))
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np
from recompute_published_tad_metrics import CONDITIONS, PEAKS
from summarize_tad_runtimes import METHODS
from tad_runtime_lock import caller_lock
from faster_cky.tad_matrix_io import load_dense_matrix
from faster_cky.deeptad_evaluation import tad_adjusted_r_squared
from faster_cky.deeptad_validation_experiment import _load_peaks
from faster_cky.tad_external_baselines import parent_domains
from faster_cky.tad_published_metrics import clustertad_quality, outermost_intervals, long_ctcf_support


def evaluate(record):
    c, chrom, method = (record[k] for k in ['condition','chromosome','method'])
    identity = fingerprint(record)
    prefix, resolution = CONDITIONS[c]
    if method == 'HiTAD':
        domains = set()
        for line in record['output_path'].read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == chrom:
                a,b = int(fields[1]), int(fields[2])
                if a % resolution or b % resolution:
                    raise ValueError('Non-aligned HiTAD output coordinates')
                domains.add((a//resolution,b//resolution))
    else:
        domains = {tuple(span) for span in read(record['output_path'])['domains']}
    domains = sorted(domains)
    run = read(record['run_path'])
    native = run.get('metrics') if method != 'HiTAD' else None
    if native is not None:
        if len(domains) != native['domains']:
            raise ValueError('Domain count differs from the evaluated output')
        # The timed runner already computes these same published metrics after
        # timing. Reuse the values tied to this output instead of recomputing.
        support = native['ctcf']
        parents = parent_domains(domains)
        if len(parents) != native['parents']:
            raise ValueError('Parent count differs from the evaluated output')
        source = run['source']
        scores = dict(r2=native['tad_adjusted_r2'], quality=native['clustertad_quality'],
                      ctcf_percent=100*support['rate'] if support['rate'] is not None else None)
        origin = 'Published metrics computed by the runner for the same domains.json'
    else:
        matrix, source = load_dense_matrix(BASE/c/'matrices'/f'{prefix}_{resolution//1000}k_KR.{chrom}', mmap=True)
        if any(not 0 <= a < b <= len(matrix) for a,b in domains):
            raise ValueError('Invalid output coordinates')
        parents = parent_domains(domains)
        fit = tad_adjusted_r_squared(matrix, domains, maximum_distance=1500000//resolution)
        quality = clustertad_quality(matrix, outermost_intervals(domains))
        quality_values = [r['quality'] for r in quality if r['quality'] is not None]
        peaks = _load_peaks(ROOT/PEAKS[c])[chrom]
        support = long_ctcf_support([(a*resolution,b*resolution) for a,b in parents],peaks)
        scores = dict(r2=float(np.mean(fit.scores)) if len(fit.scores) else None,
                      quality=float(np.mean(quality_values)) if quality_values else None,
                      ctcf_percent=100*support['rate'] if support['rate'] is not None else None)
        origin = 'Evaluation from the normalized matrix and original peak intervals'
    if any(v is not None and not np.isfinite(v) for v in scores.values()):
        raise ValueError('Nonfinite metric; no automatic publication')
    if identity != fingerprint(record):
        raise RuntimeError('Outputs changed during evaluation')
    write(record['cache_path'],dict(complete=True, fingerprint=identity, condition=c,
          method=method, chromosome=chrom, scores=scores, domains=len(domains),parents=len(parents),
          matrix=str(source),support=support,metric_origin=origin,
          unavailable_metrics=[k for k,v in scores.items() if v is None],
          updated_at=datetime.now().astimezone().isoformat(timespec='seconds')))
    print(c,method,chrom,'evaluated',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=30)
    args=parser.parse_args()
    EXTRA.mkdir(parents=True,exist_ok=True)
    with (EXTRA/'worker.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        errors={}
        while True:
            tasks=[]
            for c in CONDITIONS:
                for m in METHODS:
                    panel=records(c,m)
                    finalized=all(r['status'] in ['completed','timeout','memory_error','runtime_error'] for r in panel)
                    for r in panel:
                        if r['status']=='completed' and r['output_path'].exists() and not cached(r):
                            tasks.append((not finalized,r))
            tasks.sort(key=lambda pair:pair[0])
            for _,record in tasks:
                task='/'.join(record[k] for k in ['condition','method','chromosome'])
                identity=fingerprint(record)
                if errors.get(task,{}).get('fingerprint')==identity:continue
                write(EXTRA/'worker_status.json',dict(status='waiting_for_timing_lock',task=task,pid=os.getpid(),pending=len(tasks)))
                try:
                    if read(record['run_path']).get('metrics') is not None and record['method'] != 'HiTAD':
                        if not cached(record):evaluate(record)
                    else:
                        with caller_lock():
                            if not cached(record):evaluate(record)
                    errors.pop(task,None)
                except Exception as error:
                    errors[task]=dict(error=repr(error),fingerprint=identity)
                    print(task,repr(error),flush=True)
                write(EXTRA/'errors.json',errors)
            write(EXTRA/'worker_status.json',dict(status='watching' if args.watch else 'idle',pid=os.getpid(),errors=len(errors),
                updated_at=datetime.now().astimezone().isoformat(timespec='seconds')))
            if not args.watch:return
            time.sleep(args.poll_seconds)


if __name__=='__main__':main()
