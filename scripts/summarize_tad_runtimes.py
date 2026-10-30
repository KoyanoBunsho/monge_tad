"""Compare complete TAD pipelines; never rank unfinished attempts as fast runs."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from tad_comparison_policy import supertad_fast_enabled
from tad_memory_evidence import confirmed_memory_termination

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/tad_all_chromosomes_20261022'
PIPE=ROOT/'artifacts/tad_complete_pipeline_20261024'
EXPAND=ROOT/'artifacts/tad_method_expansion_20261024'
OUT=ROOT/'artifacts/tad_runtime_comparison_20261024'
CONDITIONS=['gm12878_25kb','gm12878_10kb','k562_25kb','imr90_25kb','gm12878_5kb','k562_5kb','imr90_5kb']
METHODS=['Ours','deepTAD','OnTAD','JOnTADS','deDoc2','RobusTAD','SpectralTAD','SuperTAD-Fast','HiTAD']
if not supertad_fast_enabled():METHODS.remove('SuperTAD-Fast')
KEYS={'OnTAD':'ontad','JOnTADS':'jontads','deDoc2':'dedoc2','SuperTAD':'supertad-binary','SuperTAD-Fast':'supertad-fast','TADtree':'tadtree','SpectralTAD':'spectraltad'}


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def normalized_status(record):
    status=record.get('status','not_run')
    return {'complete':'completed','failed':'runtime_error','memory_limit':'memory_error','acquisition_timeout':'input_unavailable'}.get(status,status)


def termination_signal(record):
    """Expose a recorded signal without guessing why the process was killed."""
    if normalized_status(record) != 'runtime_error':
        return None
    if record.get('returncode') == -9 or any(
        stage.get('returncode') == -9 and normalized_status(stage) == 'runtime_error'
        for stage in record.get('stages', [])
    ):
        return 'SIGKILL'
    return None


def status_row(condition,method,chromosome,record,path,scope='prepared_matrix_to_output'):
    signal=termination_signal(record)
    evidence=confirmed_memory_termination(path) if signal=='SIGKILL' else None
    return dict(condition=condition,method=method,chromosome=chromosome,
        status='memory_error' if evidence else normalized_status(record),
        seconds=record.get('wall_seconds',record.get('seconds',record.get('elapsed_seconds'))),
        timeout_seconds=record.get('timeout_seconds'),record=str(path.relative_to(ROOT)),scope=scope,
        termination_signal=signal,
        failure_reason='OS memory-pressure termination' if evidence else None,
        kernel_evidence=evidence['kernel_line'] if evidence else None)


def csv_write(path,rows):
    temporary=path.with_suffix(f'.{os.getpid()}.tmp')
    with temporary.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
    temporary.replace(path)


def summarize():
    OUT.mkdir(parents=True,exist_ok=True)
    pipelines={(r['condition'],r['method']):r for r in read(PIPE/'summary.json').get('rows',[])}
    summaries=[];ledger=[];diagnostic=[]
    for condition in CONDITIONS:
        for method in METHODS:
            records=[];scope='prepared_matrix_to_output';shared=0.;unit='chromosome';expected=22
            if method in ['Ours','deepTAD']:
                key='KY-W' if method=='Ours' else 'deepTAD'
                aggregate=pipelines.get((condition,key),{})
                shared=aggregate.get('boundary_seconds',0.)
                for n in range(1,23):
                    path=PIPE/condition/'status'/key/f'chr{n}.json'
                    records.append(status_row(condition,method,f'chr{n}',read(path),path,'post_boundary_pipeline'))
                boundary=read(PIPE/condition/'boundary.json')
                shared_ok=boundary.get('inference',{}).get('status')=='completed' and boundary.get('chromosomes')==[str(n) for n in range(1,23)]
            elif method=='HiTAD':
                path=OUT/'hitad'/condition/'status.json';record=read(path)
                unit='joint_22_autosomes';expected=1;shared_ok=True
                records.append(status_row(condition,method,'chr1--22',record,path))
                if condition.endswith('_5kb'):
                    oldpath=BASE/condition/'hitad/status.json';old=read(oldpath)
                    if old:diagnostic.append(dict(condition=condition,method=method,record=str(oldpath.relative_to(ROOT)),
                        seconds=old.get('seconds'),status=old.get('status'),reason='Earlier joint run included sex chromosomes; outside the exact 22-autosome comparison.'))
            else:
                shared_ok=True
                for n in range(1,23):
                    chrom=f'chr{n}'
                    key=('robustad-panel' if condition.endswith('_5kb') else 'robustad-no-panel') if method=='RobusTAD' else KEYS[method]
                    if method in ['OnTAD','JOnTADS','deDoc2'] and not condition.endswith('_5kb'):
                        path=BASE/condition/'native_callers'/chrom/'results.json'
                        rows=read(path).get('rows',[]);matching=[r for r in rows if r['method']==key]
                        record=matching[0]['run'] if matching else {}
                        if record.get('status')=='failed' and 'OutOfMemoryError' in Path(record['log']).read_text(errors='replace'):
                            record={**record,'status':'memory_error'}
                    else:
                        path=EXPAND/condition/chrom/key/'status.json';record=read(path)
                    records.append(status_row(condition,method,chrom,record,path))
                if method=='RobusTAD' and condition.endswith('_5kb'):
                    path=PIPE/condition/'robustad/boundary.status.json';boundary=read(path)
                    shared=boundary.get('seconds',0.);shared_ok=boundary.get('status')=='completed'
                    source=PIPE/condition/'robustad/boundaries.bed'
                    digest=hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None
                    for row in records:
                        original=read(ROOT/row['record'])
                        if row['status']=='completed' and original.get('boundary_sha256')!=digest:shared_ok=False
                    old=pipelines.get((condition,'RobusTAD'),{})
                    if old:diagnostic.append(dict(condition=condition,method=method,record=str((PIPE/'summary.json').relative_to(ROOT)),
                        seconds=old.get('total_seconds'),status='mixed_boundary_replay',
                        reason='Old assembly uses different stochastic boundary flags from the separately timed boundary replay; no complete-pipeline speed ratio.'))
            counts={s:sum(r['status']==s for r in records) for s in sorted({r['status'] for r in records})}
            completed=counts.get('completed',0)
            elapsed=shared+sum(r['seconds'] or 0. for r in records)
            complete=completed==expected and shared_ok
            attempted=sum(r['status'] not in ['not_run','running','preparing'] for r in records)
            summaries.append(dict(condition=condition,method=method,scope=scope,unit=unit,expected_units=expected,
                completed_units=completed,attempted_units=attempted,complete=complete,
                complete_seconds=elapsed if complete else None,observed_attempt_seconds=elapsed,
                shared_boundary_seconds=shared,timeout=counts.get('timeout',0),memory_error=counts.get('memory_error',0),
                runtime_error=counts.get('runtime_error',0),running=counts.get('running',0)+counts.get('preparing',0),
                killed=sum(r['termination_signal']=='SIGKILL' and r['status']=='runtime_error' for r in records),
                not_run=counts.get('not_run',0),launch_error=counts.get('launch_error',0),
                interrupted=counts.get('interrupted',0),
                configuration=('177-sample hg38 reference panel' if condition.endswith('_5kb') else 'author no-panel boundary mode') if method=='RobusTAD' else
                    'binary' if method=='SuperTAD' else 'h=2, discretization steps=20, window=5' if method=='SuperTAD-Fast' else
                    'S=2 Mb, N=500, M=10, p=3, q=12, gamma=500' if method=='TADtree' else '3 levels' if method=='SpectralTAD' else ''))
            ledger.extend(records)
    index={(r['condition'],r['method']):r for r in summaries}
    ratios=[]
    for c in CONDITIONS:
        ours=index[c,'Ours']['complete_seconds']
        for m in METHODS[1:]:
            previous=index[c,m]['complete_seconds']
            if ours is not None and previous is not None:
                ratios.append(dict(condition=c,method=m,ours_seconds=ours,previous_seconds=previous,
                    previous_over_ours=previous/ours,ours_over_previous=ours/previous))
    csv_write(OUT/'comparison.csv',summaries);csv_write(OUT/'chromosome_status.csv',ledger)
    csv_write(OUT/'completed_comparisons.csv',ratios)
    csv_write(OUT/'excluded_timing_records.csv',diagnostic)
    matrix=[]
    for method in METHODS:
        row={'method':method}
        for condition in CONDITIONS:
            r=index[condition,method]
            if r['complete']:
                value=f"{r['complete_seconds']:.1f}"
            else:
                value=f"pending: {r['completed_units']}/{r['expected_units']} completed"
                if r['timeout']:value+=f"; {r['timeout']} timeout"
                if r['memory_error']:value+=f"; {r['memory_error']} memory error"
                if r['killed']:value+=f"; {r['killed']} killed (SIGKILL; cause undetermined)"
                if r['runtime_error']>r['killed']:value+=f"; {r['runtime_error']-r['killed']} runtime error"
                if r['launch_error']:value+=f"; {r['launch_error']} launch error"
                if not (r['not_run'] or r['running'] or r['launch_error'] or r['interrupted']):value=value.replace('pending:','finished with failures:')
            row[condition]=value
        matrix.append(row)
    csv_write(OUT/'runtime_seconds.csv',matrix)
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    summary=dict(updated_at=now,all_comparisons_complete=all(r['complete'] for r in summaries),
        all_attempts_finished=all(sum(r[s] for s in ['not_run','running','launch_error','interrupted'])==0 for r in summaries),
        completed_method_conditions=sum(r['complete'] for r in summaries),expected_method_conditions=len(summaries),rows=summaries,ratios=ratios)
    temporary=OUT/f'summary.{os.getpid()}.tmp';temporary.write_text(json.dumps(summary,indent=2)+'\n');temporary.replace(OUT/'summary.json')
    table=[r'\begin{table}[ht]',r'\centering',r'\caption{Computation time in seconds from prepared normalized contact maps to reported domains on all 22 autosomes. Only fully completed conditions receive numerical runtime entries. The shortest time in each condition is shown in bold.}',
        r'\label{tab:all-method-runtime}',r'\footnotesize',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrrrrrrr}',r'\toprule',
        r'Method & GM25 & GM10 & K25 & IMR25 & GM5 & K5 & IMR5\\',r'\midrule']
    minimum={c:min(r['complete_seconds'] for r in summaries if r['condition']==c and r['complete']) for c in CONDITIONS}
    for m in METHODS:
        values=[m]
        for c in CONDITIONS:
            r=index[c,m]
            if r['complete']:
                value=f"{r['complete_seconds']:.1f}"
                values.append(r'\textbf{'+value+'}' if r['complete_seconds']==minimum[c] else value)
            elif r['not_run'] or r['running'] or r['launch_error'] or r['interrupted']:values.append('pending')
            else:values.append('incomplete')
        table.append(' & '.join(values)+r'\\')
    table.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    (OUT/'runtime_table.tex').write_text('\n'.join(table))
    notes=(f'Updated: {now}\n'
        f'Scope: {len(METHODS)-1} previous hierarchical TAD methods plus Ours, seven conditions, all 22 autosomes.\n'
        'Count: a complete condition needs every required execution and its shared boundary stage. Errors and timeouts stay in the ledger.\n'
        'Memory errors include SIGKILL terminations matched to macOS memory-pressure logs and an unchanged execution-record hash. Raw records remain intact. Other SIGKILL terminations retain an undetermined cause.\n'
        'Time: includes process/model startup, input reading, boundary inference, hierarchy construction, reporting, output. Excludes source download, normalization/format conversion, and evaluation.\n'
        'Input: identical normalized contact values, with caller-specific file formats. Model and size settings are explicit in comparison.csv; methods optimize different objectives.\n'
        'Limits: 1800 seconds per invocation; 20 GiB monitored caller RSS for new native runs. HiTAD learns jointly over all autosomes and uses one 1800-second genome invocation, with Docker memory limit and OOM state recorded.\n'
        'Hardware: Apple M4 Max, 36 GB; new timed TAD jobs use a shared execution lock. Separate replicate/data-acquisition jobs can still share the host. Earlier measurements also shared the host.\n'
        'Environment: deepTAD and HiTAD run Linux/amd64 code via Docker on Apple Silicon; other methods use native environments. Timings describe these implementations on this host, not hardware-independent algorithmic speed.\n'
        'Ours/deepTAD: shared measured boundary inference is charged to each method; full subsequent stages are included.\n'
        'RobusTAD: original mixed replay timings are excluded. New assembly uses the exact saved boundaries from the measured boundary stage, verified by SHA-256. The batch boundary time is charged once per condition.\n'
        'The subset of outputs used in past pilots is not extrapolated to whole-autosome totals. Progress is recorded per chromosome.\n')
    environment=read(OUT/'hitad/environment.json')
    if environment:
        notes+=f"Actual Docker VM memory for HiTAD: {environment.get('docker_vm_memory_bytes')} bytes; container upper bound {environment.get('container_memory_limit_gib')} GiB does not increase VM memory.\n"
    (OUT/'notes.txt').write_text(notes)
    print(now,'complete method/conditions',summary['completed_method_conditions'],'/',len(summaries),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        summary=summarize()
        if not args.watch or summary['all_attempts_finished']:break
        time.sleep(30)


if __name__=='__main__':main()
