"""Measure all boundary-to-output stages on all 22 selected autosomes.

The shared boundary predictor is measured once and charged to both methods.
Raw archive extraction, downloads, and accuracy scoring are outside the timing
scope. Complete process wall times include output consistency checks. Each
method starts from the same prepared matrix.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/"artifacts/tad_all_chromosomes_20261022"
OUT=ROOT/"artifacts/tad_complete_pipeline_20261024"
CONDITIONS={
    "gm12878_25kb":("HIC002",25000), "gm12878_10kb":("HIC002",10000),
    "imr90_25kb":("HIC056",25000), "k562_25kb":("HIC074",25000),
    "gm12878_5kb":("GM12878",5000), "imr90_5kb":("IMR90",5000), "k562_5kb":("K562",5000),
}
AUTOSOMES=[str(n) for n in range(1,23)]
OBJECTIVE_VARIANT="no_separator"
REFERENCE_ROOT=None


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(data,indent=2)+"\n");tmp.replace(path)


def command(args, log, timeout=1800, env=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    with log.open("w") as stream:
        stream.write(json.dumps([str(a) for a in args])+"\n");stream.flush()
        try:
            p=subprocess.run([str(a) for a in args],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
            status="completed" if p.returncode==0 else "failed";code=p.returncode
        except subprocess.TimeoutExpired:
            status="timeout";code=None
    return dict(status=status,returncode=code,seconds=time.perf_counter()-start,timeout_seconds=timeout)


def boundary(condition, chromosomes):
    import numpy as np
    from faster_cky.deeptad_monge import deeptad_boundary_patches, normalize_deeptad_patches
    from faster_cky.tad_matrix_io import load_dense_matrix
    base=OUT/condition;prefix,resolution=CONDITIONS[condition]
    result=base/"boundary.json"
    if result.exists():
        previous=json.loads(result.read_text())
        if previous.get("chromosomes")==chromosomes and previous.get("inference",{}).get("status")=="completed":
            return
        if previous.get("chromosomes")!=chromosomes and (base/"status").exists():
            (base/"status").rename(base/f"status_previous_{time.time_ns()}")
    records=[]
    for chrom in chromosomes:
        start=time.perf_counter()
        matrix,_=load_dense_matrix(OLD/condition/"matrices"/f"{prefix}_{resolution//1000}k_KR.chr{chrom}",dtype=np.float32,mmap=True)
        patches=normalize_deeptad_patches(deeptad_boundary_patches(matrix))
        (base/"patches").mkdir(parents=True,exist_ok=True)
        np.save(base/"patches"/f"chr{chrom}_patches.npy",patches)
        records.append(dict(chromosome=f"chr{chrom}",matrix_loading_and_patch_seconds=time.perf_counter()-start))
        del matrix,patches
    args=["docker","run","--rm","--platform","linux/amd64","-v",f"{ROOT}:/work","-w","/work",
        "tensorflow/tensorflow:2.9.3","python","scripts/deeptad_band_tf29_inference.py",
        "--patch-dir",(base/"patches").relative_to(ROOT),"--output-dir",(base/"predictions").relative_to(ROOT),
        "--weights","third_party/deepTAD/Model/all_27.h5","--chromosomes",*[f"chr{x}" for x in chromosomes],"--threads","1"]
    inference=command(args,base/"boundary.log")
    write(result,dict(chromosomes=chromosomes,records=records,inference=inference,
        shared_boundary_seconds=sum(r["matrix_loading_and_patch_seconds"] for r in records)+inference["seconds"],
        scope="matrix loading and patch preparation plus complete Docker invocation, model startup, prediction and intermediate output"))
    if inference["status"]!="completed":raise RuntimeError(f"boundary stage: {inference}")


def worker(condition, chrom, method):
    import numpy as np
    from dataclasses import replace
    from faster_cky.tad_matrix_io import load_dense_matrix
    from faster_cky.deeptad_filter import filter_deeptad_boundaries
    from faster_cky.deeptad_monge import atomic_tad_regions,adjacent_deeptad_cosines,public_deeptad_merge_from_similarities,distance_expected_profile
    from faster_cky.deeptad_validation_experiment import Region,Chromosome
    from faster_cky.tad_configuration import OBJECTIVE,SELECTOR
    from faster_cky.deeptad_pair_ablation import _region_costs,_select_tree_parents
    from faster_cky.deeptad_evaluation import DistanceNormalizedContacts
    from faster_cky.decoder import knuth_cky_min
    from probe_large_tad_regions import filter_band
    prefix,resolution=CONDITIONS[condition];base=OUT/condition
    start=time.perf_counter()
    matrix,_=load_dense_matrix(OLD/condition/"matrices"/f"{prefix}_{resolution//1000}k_KR.chr{chrom}",dtype=np.float64,mmap=True)
    probabilities=np.load(base/"predictions"/f"chr{chrom}_predictions.npz")["boundary_probability"]
    if resolution==5000:
        band=np.zeros((17,len(matrix)),dtype=np.float32)
        for lag in range(17):band[lag,:len(matrix)-lag]=np.diagonal(matrix,offset=lag)
        filtered=filter_band(band,probabilities)
    else:
        filtered=filter_deeptad_boundaries(matrix,np.flatnonzero(probabilities>0.5),window_size=200000//resolution,pvalue_cutoff=0.05)
    leaf_regions=atomic_tad_regions(len(matrix),filtered.filtered_indices,filtered.gap_indices,minimum_region_bins=4)
    filter_finished=time.perf_counter()
    if method=="deepTAD":
        diagonal=np.diagonal(matrix)
        nonzero=int(np.count_nonzero(matrix))-int(np.count_nonzero(diagonal))
        average=(float(np.sum(matrix))-float(np.sum(diagonal)))/nonzero if nonzero else 0.
    else:
        expected=distance_expected_profile(matrix)
        normalized=DistanceNormalizedContacts.from_contacts(matrix,maximum_distance=1500000//resolution)
        objective=replace(OBJECTIVE,duration_scale=4000000/resolution)
        if OBJECTIVE_VARIANT=="no_separator":
            objective=replace(objective,name="no_separator",probability_weight=0.,cosine_weight=0.,
                              direct_separator_weight=0.,separator_pair_weight=0.)
        selector=replace(SELECTOR,maximum_parent_bins=3000000//resolution)
    nodes=set()
    for leaves in leaf_regions:
        nodes.update(leaves)
        if len(leaves)<2:continue
        offset,stop=leaves[0][0],leaves[-1][1]
        local=[(a-offset,b-offset) for a,b in leaves]
        contacts=matrix[offset:stop,offset:stop]
        cosines=adjacent_deeptad_cosines(contacts,local)
        if method=="deepTAD":
            merged=public_deeptad_merge_from_similarities(contacts,local,cosines,average_contact=average)
            nodes.update((a+offset,b+offset) for a,b in merged)
        else:
            separators=np.asarray([b for _,b in leaves[:-1]],dtype=int)
            region=Region(tuple(leaves),probabilities[separators].astype(np.float64),cosines,frozenset())
            chromosome=Chromosome(f"chr{chrom}",matrix,(region,),np.empty((0,2),dtype=int))
            costs=_region_costs(chromosome,region,objective,expected_profile=expected)
            decoded=knuth_cky_min(costs)
            nodes.update(_select_tree_parents(region,decoded.spans,normalized,selector))
    dest=base/"output"/method/f"chr{chrom}.bed";dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text("".join(f"chr{chrom}\t{a*resolution}\t{b*resolution}\n" for a,b in sorted(nodes)))
    seconds=time.perf_counter()-start
    # The worker stopwatch stops before validation; complete-process wall times
    # in the reported totals also include these output consistency checks.
    old_method="Native deepTAD" if method=="deepTAD" else "KY-W"
    if method!="deepTAD" and OBJECTIVE_VARIANT=="no_separator":
        if REFERENCE_ROOT is None:raise ValueError("The new objective requires an independently evaluated reference output")
        checked=json.loads((REFERENCE_ROOT/"benchmark"/condition/f"chr{chrom}.json").read_text())
        if not checked.get("complete") or checked["objective"]["name"]!="no_separator":
            raise ValueError("Missing separator-free evaluation")
        reference={tuple(span) for span in checked["nodes"]["Ours"]}
    else:
        with (OLD/condition/"evaluation"/f"chr{chrom}"/"domains.tsv").open() as stream:
            reference={(int(r["start_bin"]),int(r["end_bin"])) for r in csv.DictReader(stream,delimiter="\t") if r["method"]==old_method}
    old_filtered=np.load(OLD/condition/"filtered"/f"{prefix}_{resolution//1000}k_KR.chr{chrom}.npz")
    validation=dict(output_matches=nodes==reference,domains=len(nodes),reference_domains=len(reference),
        output_difference=len(nodes^reference),boundary_matches=np.array_equal(filtered.filtered_indices,old_filtered["filtered_index"]),
        gap_matches=np.array_equal(filtered.gap_indices,old_filtered["gap_index"]))
    write(base/"worker"/method/f"chr{chrom}.json",dict(method=method,chromosome=f"chr{chrom}",objective_variant=OBJECTIVE_VARIANT,
        post_inference_seconds=seconds,load_filter_seconds=filter_finished-start,validation=validation))
    if not all(validation[k] for k in ["output_matches","boundary_matches","gap_matches"]):
        raise RuntimeError(f"Output differs from frozen benchmark: {validation}")


def post(condition,chromosomes):
    for i,chrom in enumerate(chromosomes):
        # Alternate execution order to avoid giving one method all cold reads.
        methods=["deepTAD","KY-W"] if i%2==0 else ["KY-W","deepTAD"]
        for method in methods:
            status=OUT/condition/"status"/method/f"chr{chrom}.json"
            if status.exists() and json.loads(status.read_text())["status"]=="completed":continue
            args=[sys.executable,str(Path(__file__).resolve()),"worker","--conditions",condition,"--chromosomes",chrom,"--method",method,"--output-dir",str(OUT)]
            args += ["--objective",OBJECTIVE_VARIANT]
            if REFERENCE_ROOT is not None:args += ["--reference-root",str(REFERENCE_ROOT)]
            result=command(args,OUT/condition/"logs"/method/f"chr{chrom}.log")
            write(status,result)
            print(condition,chrom,method,result["status"],round(result["seconds"],2),flush=True)


def robustad_boundary(conditions):
    accessions={"gm12878_5kb":"4DNFIXP4QG5B","imr90_5kb":"4DNFIJTOIGOI","k562_5kb":"4DNFI18UHVRO"}
    env=dict(os.environ,PYTHONPATH=str(ROOT/"third_party/RobusTAD"),ROBUSTAD_CONFIG_DIR=str(OLD/"robustad_config"))
    for condition in conditions:
        if condition not in accessions:continue
        base=OUT/condition/"robustad"
        status=base/"boundary.status.json"
        if status.exists() and json.loads(status.read_text())["status"]=="completed":continue
        args=[sys.executable,"scripts/run_robustad_command.py","lmcc","--panel",ROOT/"data/raw/robustad_hg38_5kb_50_600.pkl",
            "--chr",",".join(f"chr{n}" for n in AUTOSOMES),"--resol","5000",
            ROOT/f"data/raw/hic_neural/{accessions[condition]}.mcool",base/"boundaries"]
        write(status,command(args,base/"boundary.log",env=env))


def summarize():
    rows=[]
    for condition in CONDITIONS:
        path=OUT/condition/"boundary.json"
        if not path.exists():continue
        boundary_record=json.loads(path.read_text())
        if boundary_record["chromosomes"]!=AUTOSOMES or boundary_record["inference"]["status"]!="completed":continue
        for method in ["deepTAD","KY-W"]:
            completed=0;seconds=0.;failures=[]
            for chrom in AUTOSOMES:
                path=OUT/condition/"status"/method/f"chr{chrom}.json"
                record=json.loads(path.read_text()) if path.exists() else {"status":"not_run"}
                if record["status"]=="completed":completed+=1
                else:failures.append(f"chr{chrom}:{record['status']}")
                seconds+=record.get("seconds",0.)
            rows.append(dict(condition=condition,method=method,completed=completed,expected=22,
                boundary_seconds=boundary_record["shared_boundary_seconds"],post_inference_seconds=seconds,
                total_seconds=boundary_record["shared_boundary_seconds"]+seconds,failures=";".join(failures)))
        robust=OUT/condition/"robustad/boundary.status.json"
        if robust.exists():
            record=json.loads(robust.read_text());total=record["seconds"];completed=0;failures=[]
            if record["status"]!="completed":
                raise RuntimeError(f"Incomplete 22-autosome RobusTAD boundary timing: {condition}")
            chroms={f"chr{n}" for n in AUTOSOMES}
            def boundary_rows(path):
                rows={}
                for line in path.read_text().splitlines():
                    fields=line.split()
                    if fields and fields[0] in chroms:
                        rows[tuple(fields[:3])]=tuple(float(v) for v in fields[3:])
                return rows
            new_rows=boundary_rows(OUT/condition/"robustad/boundaries.bed")
            old_rows=boundary_rows(OLD/condition/"robustad/boundaries.bed")
            same_keys=new_rows.keys()==old_rows.keys()
            same_scores=same_keys and all(new_rows[k][::2]==old_rows[k][::2] for k in new_rows)
            changed=sum(new_rows[k][1::2]!=old_rows[k][1::2] for k in new_rows) if same_keys else None
            write(OUT/condition/"robustad/boundary_replay_validation.json",dict(
                coordinate_rows_equal=same_keys,contact_scores_equal=same_scores,
                rows_with_changed_boundary_flags=changed,
                interpretation="Boundary calling uses unseeded shuffled decoys. The fresh timing run may have different binary boundary flags. Assembly/cleaning use the original frozen boundary file; totals are separately measured stage sums, not one output-identical execution."))
            if not same_scores:
                raise RuntimeError(f"RobusTAD coordinates or contact scores changed beyond stochastic flags: {condition}")
            for chrom in AUTOSOMES:
                base=OLD/condition/"robustad"/f"chr{chrom}"
                a=json.loads((base/"assembly.status.json").read_text())
                c=json.loads((base/"clean.status.json").read_text()) if (base/"clean.status.json").exists() else {}
                total+=a["seconds"]+c.get("seconds",0.)
                if a["status"]==c.get("status")=="completed":completed+=1
                else:failures.append(f"chr{chrom}:{a['status']}")
            rows.append(dict(condition=condition,method="RobusTAD",completed=completed,expected=22,
                boundary_seconds=record["seconds"],post_inference_seconds=total-record["seconds"],total_seconds=total,
                failures=";".join(failures)))
    write(OUT/"summary.json",dict(rows=rows,
        scope="Sum of measured stages from prepared normalized matrices to domain files, including loading, model/process startup, boundary filtering, contact features, assembly, reporting, output and output consistency checks; excludes archive extraction, downloads and metric evaluation.",
        robustad_scope="Fresh 22-autosome LMCC timing plus existing chromosome-level assembly and clean timings using the original frozen boundary file. The randomized boundary caller changes some flags on replay. These are independently measured stage sums, including censored timeouts, not one completed output-identical run."))
    if rows:
        with (OUT/"summary.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows),flush=True)


def main():
    global OUT,OBJECTIVE_VARIANT,REFERENCE_ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["boundary","post","worker","robustad","summarize","all"])
    parser.add_argument("--conditions",nargs="+",choices=list(CONDITIONS),default=list(CONDITIONS))
    parser.add_argument("--chromosomes",nargs="+",default=AUTOSOMES)
    parser.add_argument("--method",choices=["deepTAD","KY-W"],default="KY-W")
    parser.add_argument("--force-post",action="store_true",help="Archive previous stage statuses and repeat post-inference timing")
    parser.add_argument("--output-dir",type=Path,default=OUT)
    parser.add_argument("--objective",choices=["full","no_separator"],default="no_separator")
    parser.add_argument("--reference-root",type=Path)
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    OBJECTIVE_VARIANT=args.objective
    REFERENCE_ROOT=args.reference_root.resolve() if args.reference_root else None
    os.environ.update(PYTHONPATH=str(ROOT/"src"),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MPLCONFIGDIR=str(ROOT/".cache/matplotlib"))
    sys.path.insert(0,str(ROOT/"src"));OUT.mkdir(parents=True,exist_ok=True)
    if args.phase=="worker":worker(args.conditions[0],args.chromosomes[0],args.method);return
    for condition in args.conditions:
        if args.force_post and (OUT/condition/"status").exists():
            (OUT/condition/"status").rename(OUT/condition/f"status_previous_{time.time_ns()}")
        if args.phase in ["boundary","all"]:boundary(condition,args.chromosomes)
        if args.phase in ["post","all"]:post(condition,args.chromosomes)
    if args.phase=="robustad" or (args.phase=="all" and args.chromosomes==AUTOSOMES):robustad_boundary(args.conditions)
    if args.phase in ["summarize","all"]:summarize()


if __name__=="__main__":main()
