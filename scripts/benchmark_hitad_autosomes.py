"""Time HiTAD's joint learning on exactly the 22 benchmark autosomes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import pandas as pd
import cooler
import h5py
from faster_cky.tad_matrix_io import load_dense_matrix
from faster_cky.process_control import run_with_timeout
from recompute_published_tad_metrics import BASE,CONDITIONS,write,read
from tad_runtime_lock import caller_lock

OUT=ROOT/'artifacts/tad_runtime_comparison_20261024/hitad'


def prepare(condition,path):
    prefix,res=CONDITIONS[condition];bins=[];sources=[];offset=0
    for n in range(1,23):
        matrix,source=load_dense_matrix(BASE/condition/'matrices'/f'{prefix}_{res//1000}k_KR.chr{n}',mmap=True)
        size=len(matrix);del matrix
        bins.append(pd.DataFrame(dict(chrom=f'chr{n}',start=np.arange(size)*res,end=(np.arange(size)+1)*res)))
        sources.append(dict(source=str(source),chromosome=f'chr{n}',bins=size,offset=offset));offset+=size
    if path.exists():return sources
    def pixels():
        for item in sources:
            matrix,_=load_dense_matrix(item['source'],mmap=True)
            for start in range(0,len(matrix),64):
                block=np.asarray(matrix[start:start+64]);u,v=np.nonzero(block);keep=v>=u+start;u,v=u[keep],v[keep]
                yield pd.DataFrame(dict(bin1_id=u+start+item['offset'],bin2_id=v+item['offset'],count=block[u,v]))
            del matrix
    partial=path.with_suffix('.partial.cool')
    cooler.create_cooler(str(partial),pd.concat(bins,ignore_index=True),pixels(),ordered=True,dtypes={'count':np.float64})
    with h5py.File(partial,'r+') as handle:handle['bins'].create_dataset('weight',data=np.ones(offset))
    partial.replace(path)
    write(path.with_suffix('.provenance.json'),dict(sources=sources,normalization='Unchanged normalized values, identity weights',
        chromosomes=[f'chr{n}' for n in range(1,23)],preparation_in_timing=False))
    return sources


def run(condition,args):
    work=OUT/condition;work.mkdir(parents=True,exist_ok=True);status=work/'status.json'
    if read(status).get('status') in ['completed','timeout','memory_error','runtime_error']:return
    write(status,dict(status='preparing',condition=condition,pid=os.getpid()))
    source=work/'input.cool';sources=prepare(condition,source)
    container_work='/work/'+str(work.relative_to(ROOT));res=CONDITIONS[condition][1]
    (work/'datasets.txt').write_text(f'res:{res}\nrep1:{container_work}/input.cool\n')
    name='tad-runtime-'+condition.replace('_','-')
    command=['docker','run','--init','--name',name,'--platform','linux/amd64','--memory',f'{args.memory_gib}g',
             '-v',f'{ROOT}:/work','-w',container_work,'tad-hitad:20261022','python','/work/scripts/run_hitad_command.py',
             '-d','datasets.txt','-O','domains.bed','--maxsize','4000000','--cpu-core','1','--exclude','--logFile','hitad.log']
    record=dict(status='running',condition=condition,method='HiTAD',command=command,
        timeout_seconds=args.timeout_seconds,memory_limit_gib=args.memory_gib,execution_unit='joint_22_autosomes',
        chromosomes=[f'chr{n}' for n in range(1,23)],scope='Docker/model startup, matrix loading, HMM learning, all-autosome calls, output; prepared Cooler input',
        sources=sources,pid=os.getpid())
    write(status,record);started=time.monotonic()
    with (work/'driver.log').open('w') as log:
        try:
            result=run_with_timeout(command,timeout=args.timeout_seconds,stdout=log,stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode,status='completed' if result.returncode==0 and (work/'domains.bed').exists() else 'runtime_error')
        except subprocess.TimeoutExpired:
            record.update(returncode=None,status='timeout')
            subprocess.run(['docker','stop','-t','1',name],stdout=log,stderr=subprocess.STDOUT,timeout=30)
        finally:
            record['wall_seconds']=time.monotonic()-started
    inspected=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
    if inspected.returncode==0:
        state=json.loads(inspected.stdout)[0]['State'];record['container_state']=state
        if state.get('OOMKilled'):record['status']='memory_error'
        subprocess.run(['docker','rm',name],stdout=subprocess.DEVNULL,check=True)
    else:
        log=(work/'driver.log').read_text(errors='replace')
        if any(message in log.lower() for message in ['permission denied','cannot connect to the docker daemon','connection refused','no such image']):
            record['status']='launch_error'
            record['error']='Container could not start; this is not a HiTAD runtime failure.'
    write(status,record)
    print(condition,record['status'],record['wall_seconds'],flush=True)
    source.unlink(missing_ok=True)


def main():
    global OUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timeout-seconds',type=float,default=1800)
    p.add_argument('--memory-gib',type=int,default=20)
    p.add_argument('--conditions',nargs='+',choices=list(CONDITIONS),default=list(CONDITIONS))
    p.add_argument('--output-dir',type=Path,default=OUT)
    args=p.parse_args()
    OUT=args.output_dir.resolve()
    # Verify the container runtime before spending time exporting entire genomes.
    info=subprocess.run(['docker','info','--format','{{json .}}'],capture_output=True,text=True)
    if info.returncode:
        raise RuntimeError('Docker launch permission or daemon unavailable: '+info.stderr)
    environment=json.loads(info.stdout)
    write(OUT/'environment.json',dict(docker_architecture=environment.get('Architecture'),
        docker_vm_memory_bytes=environment.get('MemTotal'),container_memory_limit_gib=args.memory_gib,
        requested_platform='linux/amd64',host='Apple Silicon; Docker uses emulation',
        note='The Docker VM memory is smaller than the native-process budget; a HiTAD memory failure is not evidence of slower execution.'))
    for condition in args.conditions:
        with caller_lock():run(condition,args)


if __name__=='__main__':main()
