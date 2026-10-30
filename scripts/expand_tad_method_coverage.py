"""Fill previously unrun method/condition combinations on all 22 autosomes.

Keep the new records separate from historical runs. By default run one caller at a time,
with 1800 seconds per execution and an explicit 20 GiB resident-memory budget.
RobusTAD uses its author-supported no-panel mode for hg19 10/25 kb inputs;
the released hg38 5 kb reference panel is not transferred across assemblies.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                  MPLCONFIGDIR=str(ROOT/'.cache/matplotlib-tad-expansion'),
                  PYTHONPATH=str(ROOT/'third_party/RobusTAD')+os.pathsep+str(ROOT/'src'))
import numpy as np
from faster_cky.process_control import run_with_rss_limit,MemoryLimitExceeded
from faster_cky.tad_external_baselines import _method_command,parse_ontad,parse_dedoc2,parse_jontads,parse_supertad,parent_domains
from faster_cky.tad_matrix_io import load_dense_matrix
from faster_cky.deeptad_evaluation import tad_adjusted_r_squared
from faster_cky.deeptad_validation_experiment import _load_peaks
from faster_cky.tad_published_metrics import clustertad_quality,outermost_intervals,long_ctcf_support
from recompute_published_tad_metrics import BASE,CONDITIONS,PEAKS,read
from tad_runtime_lock import caller_lock
from tad_comparison_policy import supertad_fast_enabled

OUT=ROOT/'artifacts/tad_method_expansion_20261024'
METHODS=['ontad','jontads','dedoc2','robustad-no-panel','robustad-panel']
ADDITIONAL=['spectraltad','supertad-fast']
SUPERTAD_FAST_IMPLEMENTATION=ROOT/'artifacts/tad_supertad_fast_campaign_20261026/implementation'
BASELINE_REPAIRS=ROOT/'artifacts/tad_baseline_retries_20261026/implementation.json'
TERMINAL={'completed','timeout','memory_error','runtime_error'}


def write(path, data):
    """Use unique temporary files because independent jobs update the ledger."""
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f'{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def required(condition):
    additional=[m for m in ADDITIONAL if m!='supertad-fast' or supertad_fast_enabled()]
    return (METHODS[:3]+['robustad-panel'] if condition.endswith('_5kb') else ['robustad-no-panel'])+additional


def ledger():
    rows=[]
    for c in CONDITIONS:
        for n in range(1,23):
            for m in required(c):
                p=OUT/c/f'chr{n}'/m/'status.json'
                d=read(p)
                rows.append(dict(condition=c,chromosome=f'chr{n}',method=m,
                                 status=d.get('status','not_run'),record=str(p.relative_to(ROOT))))
    write(OUT/'coverage.json',dict(complete=all(r['status'] in TERMINAL for r in rows),
        expected=len(rows),counts={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},rows=rows))


def execute(command,work,name,args):
    path=work/f'{name}.json'
    if path.exists() and read(path).get('status') in ['completed','timeout','memory_error','runtime_error']:
        return read(path)
    if path.exists():
        old=read(path);old.update(status='interrupted',reason='Resumed unfinished execution; previous attempt retained')
        stamp=str(time.time_ns());write(work/f'{name}.{stamp}.previous.json',old)
        log=work/f'{name}.log'
        if log.exists():log.rename(work/f'{name}.{stamp}.previous.log')
    start=time.monotonic();record=dict(command=list(map(str,command)),timeout_seconds=args.timeout_seconds,
                                      execution_mode='parallel' if getattr(args,'accuracy_parallel',False) else 'serial',
                                      memory_limit_bytes=args.memory_gib*1024**3,status='running')
    write(path,record)
    with (work/f'{name}.log').open('w') as log:
        try:
            result,rss=run_with_rss_limit(record['command'],timeout=args.timeout_seconds,
                memory_bytes=record['memory_limit_bytes'],stdout=log,stderr=subprocess.STDOUT)
            record.update(status='completed' if result.returncode==0 else 'runtime_error',
                          returncode=result.returncode,peak_rss_bytes=rss)
        except subprocess.TimeoutExpired:
            record.update(status='timeout',returncode=None)
        except MemoryLimitExceeded as e:
            record.update(status='memory_error',returncode=None,peak_rss_bytes=e.observed)
        except OSError as e:
            record.update(status='launch_error',returncode=None,error=repr(e))
    record['wall_seconds']=time.monotonic()-start
    # deDoc2 catches Java exceptions and can still return zero and write an
    # empty output. A zero exit code alone is not a successful prediction.
    log_text=(work/f'{name}.log').read_text(errors='replace')
    if record['status']=='completed':
        if any(marker in log_text for marker in ['OutOfMemoryError','MemoryError','Cannot allocate memory','cannot allocate vector']):
            record.update(status='memory_error',reason='Memory failure reported in execution log')
        elif 'java.lang.NumberFormatException' in log_text:
            record.update(status='runtime_error',reason='Input parsing failed despite zero exit code')
    write(path,record)
    return record


def export_text(matrix,path,delimiter=' '):
    if path.exists():return
    start=time.monotonic();temporary=path.with_suffix('.partial')
    digest=hashlib.sha256()
    with temporary.open('w') as out:
        for i in range(0,len(matrix),64):
            block=np.asarray(matrix[i:i+64])
            if not np.all(np.isfinite(block)):raise ValueError('Nonfinite caller input')
            digest.update(np.ascontiguousarray(block).tobytes())
            np.savetxt(out,block,fmt='%.17g',delimiter=delimiter)
    temporary.replace(path)
    write(path.with_suffix('.export.json'),dict(shape=list(matrix.shape),dtype=str(matrix.dtype),
        unmodified_array_sha256=digest.hexdigest(),format='%.17g; row-streamed, no renormalization',delimiter=delimiter,
        seconds=time.monotonic()-start))


def archive_input_failure(work):
    """Preserve the failed attempt before retrying with the corrected input."""
    archive=work/'previous_attempts'/f'input_format_{time.time_ns()}'
    archive.mkdir(parents=True)
    for path in list(work.iterdir()):
        if path.is_file():path.replace(archive/path.name)
    return archive


def export_cool(matrix,path,chrom,res):
    if path.exists():return
    import cooler,pandas as pd,h5py
    n=len(matrix)
    bins=pd.DataFrame(dict(chrom=[chrom]*n,start=np.arange(n)*res,end=(np.arange(n)+1)*res))
    def pixels():
        for i in range(0,n,64):
            block=np.asarray(matrix[i:i+64]);u,v=np.nonzero(block)
            keep=v>=u+i;u,v=u[keep],v[keep]
            yield pd.DataFrame(dict(bin1_id=u+i,bin2_id=v,count=block[u,v]))
    temporary=path.with_suffix('.partial.cool')
    cooler.create_cooler(str(temporary),bins,pixels(),ordered=True,dtypes={'count':np.float64})
    with h5py.File(temporary,'r+') as handle:handle['bins'].create_dataset('weight',data=np.ones(n))
    temporary.replace(path)
    write(path.with_suffix('.export.json'),dict(source='Original normalized dense matrix, upper triangle',
        balance='Identity weights; no additional balancing',shape=list(matrix.shape)))


def run_one(c,chrom,method,args):
    prefix,res=CONDITIONS[c];work=OUT/c/chrom/method;work.mkdir(parents=True,exist_ok=True)
    target=work/'status.json';previous=read(target)
    retry_archive=None
    if previous.get('retryable_input_format_error'):
        retry_archive=archive_input_failure(work)
    elif previous.get('status') in ['completed','timeout','memory_error','runtime_error']:
        return
    mode='parallel' if getattr(args,'accuracy_parallel',False) else 'serial'
    starting=dict(status='running',condition=c,chromosome=chrom,method=method,pid=os.getpid(),execution_mode=mode)
    # Preserve the archived failure and implementation identity through the retry.
    if previous.get('retry'):
        starting['retry']=previous['retry']
    repair=read(BASELINE_REPAIRS).get('methods',{}).get(method)
    if repair:
        for relative,digest in repair['source_sha256'].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Baseline repair changed without validation: {relative}')
        starting['implementation_repair']=repair
    if method=='supertad-fast':
        starting['target_binary_sha256']=read(SUPERTAD_FAST_IMPLEMENTATION/'manifest.json')['binary_sha256']
    write(target,starting)
    ledger()
    matrix,source=load_dense_matrix(BASE/c/'matrices'/f'{prefix}_{res//1000}k_KR.{chrom}',mmap=True)
    # Parallel jobs own their converted inputs; no job can remove a sibling's input.
    shared=work if getattr(args,'accuracy_parallel',False) else OUT/c/chrom
    stages=[]
    if method.startswith('robustad-'):
        output=work/'tads'
        python=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_robustad_command.py')]
        if method=='robustad-panel':
            from run_tad_all_chromosomes import SOURCES
            accession=SOURCES[c.split('_')[0]][0]
            input_path=ROOT/'data/raw/hic_neural'/f'{accession}.mcool'
            boundary=getattr(args,'boundary_root',ROOT/'artifacts/tad_complete_pipeline_20261024')/c/'robustad/boundaries'
            assert boundary.with_suffix('.bed').exists(),boundary
            commands=[]
        else:
            input_path=shared/'normalized.cool';export_cool(matrix,input_path,chrom,res)
            boundary=work/'boundaries'
            commands=[('boundary',python+['boundary','--chr',chrom,'--resol',str(res),str(input_path),str(boundary)])]
        commands += [('assembly',python+['assembly','--chr',chrom,'--resol',str(res),'--mintad','50000','--maxtad','3000000',str(input_path),str(boundary)+'.bed',str(output)]),
          ('clean',python+['clean','--resol',str(res),str(output)+'_hq.bed',str(boundary)+'.bed',str(input_path)])]
        for name,command in commands:
            result=execute(command,work,name,args);stages.append(result)
            if result['status']!='completed':break
        output_path=work/'tads_hq_cleaned.bed'
        spans=[]
        if stages[-1]['status']=='completed' and output_path.exists():
            for line in output_path.read_text().splitlines():
                f=line.split()
                if f and f[0]==chrom:spans.append((int(f[1])//res,int(f[2])//res))
    elif method=='spectraltad':
        input_path=source
        if source.suffix!='.npy':
            input_path=shared/'matrix.npy'
            if not input_path.exists():np.save(input_path,matrix)
        output_path=work/'domains.tsv'
        os.environ['R_LIBS_USER']=str(ROOT/'.cache/Rlib')
        stages=[execute(['Rscript',ROOT/'scripts/run_spectraltad.R',input_path,chrom,str(res),'3',output_path],work,'caller',args)]
        spans=[]
        if stages[-1]['status']=='completed' and output_path.exists():
            for line in output_path.read_text().splitlines():
                f=line.split()
                if f:spans.append((int(f[1])//res,int(f[2])//res))
    elif method=='tadtree':
        input_path=shared/'matrix.txt';export_text(matrix,input_path)
        output_path=work/'caller_output'/chrom/'N500.txt'
        (work/'caller_output').mkdir(exist_ok=True)
        control=work/'control.txt'
        control.write_text(f'S = {2000000//res}\nM = 10\np = 3\nq = 12\ngamma = 500\ncontact_map_path = {input_path}\ncontact_map_name = {chrom}\nN = 500\noutput_directory = {work}/caller_output\n')
        stages=[execute([ROOT/'.venv/bin/python',ROOT/'.cache/tadtree_py3/TADtree.py',control],work,'caller',args)]
        spans=[]
        if stages[-1]['status']=='completed':
            outputs=sorted((work/'caller_output'/chrom).glob('N*.txt'),key=lambda p:int(p.stem[1:]))
            if outputs:
                output_path=outputs[-1]
                for line in output_path.read_text().splitlines()[1:]:
                    f=line.split()
                    if len(f)>=3:spans.append((int(f[1]),int(f[2])))
    else:
        # The author's dense-matrix reader splits on tabs, not whitespace.
        input_path=shared/('matrix.dedoc2.tsv' if method=='dedoc2' else 'matrix.txt')
        export_text(matrix,input_path,delimiter='\t' if method=='dedoc2' else ' ')
        if method.startswith('supertad'):
            # Author code derives output names from the input path, even with -w.
            link=work/'matrix.txt'
            if not link.exists() and not link.is_symlink():link.symlink_to(input_path)
            if method=='supertad-multi':
                command=[str(args.supertad_binary),'multi',str(link),'-h','2','--chrom1',chrom,'-r',str(res),'--maxbin',str(args.maximum_tad_bp//res)]
                output_path=work/'matrix.txt.multi.tsv'
            else:command,output_path=_method_command(method,link,work/'domains',chrom,res,args)
        else:command,output_path=_method_command(method,input_path,work/'domains',chrom,res,args)
        stages=[execute(command,work,'caller',args)]
        spans=[]
        if stages[-1]['status']=='completed' and output_path.exists():
            parser=parse_supertad if method.startswith('supertad') else {'ontad':parse_ontad,'jontads':parse_jontads,'dedoc2':parse_dedoc2}[method]
            spans=parser(output_path,len(matrix))
            low=work/'domains.window.TAD'
            if method=='dedoc2' and low.exists():spans=sorted(set(spans+parse_dedoc2(low,len(matrix))))
    status=stages[-1]['status']
    if status=='completed' and not output_path.exists():status='runtime_error'
    record=dict(status=status,condition=c,chromosome=chrom,method=method,stages=stages,
                source=str(source),matrix_bins=len(matrix),output=str(output_path),resolution=res,
                wall_seconds=sum(s['wall_seconds'] for s in stages),execution_mode=mode,
                timing_provisional=mode=='parallel')
    for key in ('retry','implementation_repair'):
        if key in starting:record[key]=starting[key]
    if retry_archive is not None:
        record['previous_attempt_archive']=str(retry_archive.relative_to(ROOT))
    if method=='robustad-panel':
        record['shared_boundary_record']=str(boundary.parent/'boundary.status.json')
        record['boundary_sha256']=hashlib.sha256(boundary.with_suffix('.bed').read_bytes()).hexdigest()
    if method=='supertad-fast':
        record['implementation']=read(SUPERTAD_FAST_IMPLEMENTATION/'manifest.json')
    if status=='runtime_error':
        logs='\n'.join(p.read_text(errors='replace')[-5000:] for p in work.glob('*.log'))
        if any(x in logs for x in ['OutOfMemoryError','MemoryError','Cannot allocate memory','cannot allocate vector']):record['status']='memory_error'
    if status=='completed' and not getattr(args,'skip_metrics',False):
        spans=sorted(set(spans));parents=parent_domains(spans)
        peaks=_load_peaks(ROOT/PEAKS[c])[chrom]
        quality=clustertad_quality(matrix,outermost_intervals(spans))
        scores=[v['quality'] for v in quality if v['quality'] is not None]
        fit=tad_adjusted_r_squared(matrix,spans,maximum_distance=1500000//res)
        record['metrics']=dict(domains=len(spans),parents=len(parents),
            tad_adjusted_r2=float(np.mean(fit.scores)),clustertad_quality=float(np.mean(scores)) if scores else None,
            ctcf=long_ctcf_support([(a*res,b*res) for a,b in parents],peaks))
        write(work/'domains.json',dict(domains=spans,parents=parents))
    write(target,record);ledger();print(c,chrom,method,status,flush=True)
    del matrix
    # Keep only one derived text input at a time; original matrices stay intact.
    selected=[m for m in required(c) if m in args.methods]
    if getattr(args,'accuracy_parallel',False) or all(read(shared/m/'status.json').get('status') in TERMINAL for m in selected):
        for temporary in [shared/'matrix.txt',shared/'matrix.dedoc2.tsv',shared/'matrix.npy',shared/'normalized.cool']:
            temporary.unlink(missing_ok=True)


def main():
    global OUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timeout-seconds',type=float,default=1800)
    p.add_argument('--memory-gib',type=int,default=20)
    p.add_argument('--conditions',nargs='+',choices=list(CONDITIONS),default=list(CONDITIONS))
    p.add_argument('--chromosomes',nargs='+',type=int,default=list(range(22,0,-1)))
    p.add_argument('--methods',nargs='+',choices=METHODS+ADDITIONAL,default=METHODS+ADDITIONAL)
    p.add_argument('--batch-lock',action='store_true',help='Hold the shared timing lock for this batch; still run callers one at a time.')
    p.add_argument('--accuracy-parallel',action='store_true',help='Run under the memory-aware accuracy scheduler; mark times provisional.')
    p.add_argument('--output-root',type=Path,default=OUT)
    p.add_argument('--boundary-root',type=Path,default=ROOT/'artifacts/tad_complete_pipeline_20261024')
    p.add_argument('--repeat-all-methods',action='store_true',help='Also repeat existing native-method comparisons at 10/25 kb.')
    p.add_argument('--skip-metrics',action='store_true',help='Remeasure computation time without recomputing accuracy.')
    args=p.parse_args()
    if not supertad_fast_enabled() and 'supertad-fast' in args.methods:
        args.methods=[m for m in args.methods if m!='supertad-fast']
        print('SuperTAD-Fast excluded under the recorded experiment-time budget.',flush=True)
    if args.batch_lock and args.accuracy_parallel:p.error('Batch timing lock and parallel accuracy execution are mutually exclusive.')
    OUT=args.output_root.resolve()
    args.maximum_tad_bp=10000000
    args.ontad_binary=ROOT/'third_party/OnTAD/src/OnTAD'
    args.dedoc2_jar=ROOT/'third_party/deDoc2/deDoc2.jar'
    args.jontads_python=ROOT/'third_party/venvs/jontads/bin/python'
    args.jontads_script=ROOT/'third_party/JOnTADS/JOnTADS/JOnTADS.py'
    args.supertad_binary=ROOT/'third_party/SuperTAD/build/SuperTAD'
    args.supertad_fast_binary=SUPERTAD_FAST_IMPLEMENTATION/'SuperTAD'
    args.supertad_steps=20
    if 'supertad-fast' in args.methods:
        implementation=read(SUPERTAD_FAST_IMPLEMENTATION/'manifest.json')
        if not args.supertad_fast_binary.exists() or hashlib.sha256(args.supertad_fast_binary.read_bytes()).hexdigest()!=implementation.get('binary_sha256'):
            raise SystemExit('The validated SuperTAD-Fast build is missing or has changed.')
    OUT.mkdir(parents=True,exist_ok=True);ledger()
    write(OUT/'protocol.json',dict(conditions=CONDITIONS,expected_autosomes=list(range(1,23)),
        timeout_seconds_per_execution=args.timeout_seconds,memory_limit_gib=args.memory_gib,
        run_order='All 22 included; independent accuracy jobs may execute concurrently; their times are provisional.',
        robustad='Author no-panel boundary command for hg19 10/25 kb; official hg38 5 kb panel is inapplicable. Report as a separate configuration.',
        additional_methods=ADDITIONAL,
        parameters={'SpectralTAD':'3 levels','SuperTAD-Fast':'h=2, discretization steps=20, neighbor window=5, maximum domain span=10 Mb'},
        supertad_fast_implementation=read(SUPERTAD_FAST_IMPLEMENTATION/'manifest.json'),
        inputs='Unchanged normalized matrices from the existing benchmark; streamed text or Cooler with identity weights.'))
    with caller_lock() if args.batch_lock else nullcontext():
        for n in args.chromosomes:
            for c in args.conditions:
                for m in (args.methods if args.repeat_all_methods else required(c)):
                    if m in args.methods:
                        with nullcontext() if args.batch_lock or args.accuracy_parallel else caller_lock():
                            run_one(c,f'chr{n}',m,args)


if __name__=='__main__':main()
