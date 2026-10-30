"""Re-evaluate fixed domain outputs using ClusterTAD and Long et al. metrics."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
import numpy as np
from faster_cky.tad_published_metrics import clustertad_quality, long_ctcf_support, outermost_intervals
from faster_cky.deeptad_validation_experiment import _load_peaks
from faster_cky.tad_external_baselines import parent_domains, parse_ontad, parse_jontads, parse_dedoc2
from faster_cky.tad_matrix_io import load_dense_matrix

BASE=ROOT/"artifacts/tad_all_chromosomes_20261022"
OUT=ROOT/"artifacts/tad_metric_revision_20261024"
CONDITIONS={"gm12878_25kb":("HIC002",25000),"gm12878_10kb":("HIC002",10000),
 "imr90_25kb":("HIC056",25000),"k562_25kb":("HIC074",25000),
 "gm12878_5kb":("GM12878",5000),"imr90_5kb":("IMR90",5000),"k562_5kb":("K562",5000)}
PEAKS={"gm12878_25kb":"data/raw/deeptad_functional/GSM749704_GM12878_CTCF_hg19.narrowPeak.gz",
 "gm12878_10kb":"data/raw/deeptad_functional/GSM749704_GM12878_CTCF_hg19.narrowPeak.gz",
 "k562_25kb":"data/raw/tad_metric_validation_20261024/ENCFF738TKN.bed.gz",
 "imr90_25kb":"data/raw/tad_metric_validation_20261024/ENCFF453XKM.bed.gz",
 "gm12878_5kb":"data/raw/tad_metric_validation_20261024/ENCFF796WRU.bed.gz",
 "imr90_5kb":"data/raw/tad_metric_validation_20261024/ENCFF203SRF.bed.gz",
 "k562_5kb":"data/raw/tad_metric_validation_20261024/ENCFF901CBP.bed.gz"}


def read(p):return json.loads(p.read_text()) if p.exists() else {}
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(d,indent=2)+"\n");tmp.replace(p)
def mean(xs):
 values=[x for x in xs if x is not None and np.isfinite(x)]
 return statistics.mean(values) if values else None
def csv_out(name,rows):
 if not rows:return
 with (OUT/name).open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def reported(condition,chrom,include_greedy=False):
 base=BASE/condition;res=CONDITIONS[condition][1]
 domains,parents=defaultdict(set),defaultdict(set)
 aliases={"Native deepTAD":"deepTAD","KY-W":"Ours"}
 if include_greedy:aliases["Greedy-W"]="Greedy"
 with (base/"evaluation"/chrom/"domains.tsv").open() as stream:
  for r in csv.DictReader(stream,delimiter="\t"):
   if r["method"] not in aliases:continue
   method=aliases[r["method"]];span=(int(r["start_bin"]),int(r["end_bin"]))
   domains[method].add(span)
   if r["is_parent"]=="True":parents[method].add(span)
 for row in read(base/"native_callers"/chrom/"results.json").get("rows",[]):
  if row["run"]["status"]!="completed":continue
  parser={"ontad":parse_ontad,"jontads":parse_jontads,"dedoc2":parse_dedoc2}[row["method"]]
  method={"ontad":"OnTAD","jontads":"JOnTADS","dedoc2":"deDoc2"}[row["method"]]
  spans=parser(Path(row["output"]),row["matrix_bins"])
  if row["method"]=="dedoc2":
   output=Path(row["output"])
   window=output.with_name(output.stem+".window.TAD")
   if window.exists():spans=sorted(set(spans+parse_dedoc2(window,row["matrix_bins"])))
  domains[method].update(spans);parents[method].update(parent_domains(spans))
 robust=base/"robustad"/chrom
 if read(robust/"assembly.status.json").get("status")==read(robust/"clean.status.json").get("status")=="completed":
  spans=[]
  for line in (robust/"tads_hq_cleaned.bed").read_text().splitlines():
   f=line.split()
   if f and f[0]==chrom:spans.append((int(f[1])//res,int(f[2])//res))
  domains["RobusTAD"].update(spans);parents["RobusTAD"].update(parent_domains(spans))
 for method in domains:parents.setdefault(method,set())
 return domains,parents


def ctcf():
 rows=[];provenance={}
 for c,filename in PEAKS.items():
  path=ROOT/filename
  meta=OUT/"reference_sources"/(path.name.split(".")[0]+".json")
  if meta.exists():
   expected=read(meta)["md5sum"]
   if hashlib.md5(path.read_bytes()).hexdigest()!=expected:raise ValueError(f"ENCODE checksum mismatch: {path}")
  provenance[c]=dict(file=filename,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  peaks=_load_peaks(path);res=CONDITIONS[c][1]
  for n in range(1,23):
   chrom=f"chr{n}";_,parents=reported(c,chrom)
   if chrom not in peaks:raise ValueError(f"Missing peak chromosome: {c} {chrom}")
   for method,spans in parents.items():
    rows.append(dict(condition=c,chromosome=chrom,method=method,
      **long_ctcf_support([(a*res,b*res) for a,b in spans],peaks[chrom])))
 summaries=[]
 for c,method in sorted({(r["condition"],r["method"]) for r in rows}):
  selected=[r for r in rows if (r["condition"],r["method"])==(c,method)]
  summaries.append(dict(condition=c,method=method,chromosomes=len(selected),
   mean_rate=mean(r["rate"] for r in selected),supported=sum(r["supported"] for r in selected),
   evaluated=sum(r["evaluated"] for r in selected)))
 write(OUT/"ctcf.json",dict(complete=all(sum(r["condition"]==c and r["method"]==m for r in rows)==22 for c in PEAKS for m in ["deepTAD","Ours"]),
  tolerance_bp=10000,reference="10.1186/s13072-022-00436-9",rows=rows,summary=summaries,tracks=provenance))
 csv_out("ctcf_per_chromosome.csv",rows);csv_out("ctcf_summary.csv",summaries)
 print("CTCF recomputation complete",len(rows),flush=True)


def quality(conditions,chromosomes):
 for c in conditions:
  prefix,res=CONDITIONS[c]
  for n in chromosomes:
   chrom=f"chr{n}";target=OUT/"quality"/c/f"{chrom}.json"
   cached=read(target)
   if cached and (cached.get("output_definition_version")==2 or "deDoc2" not in cached.get("methods",{})):continue
   start=time.monotonic()
   matrix,path=load_dense_matrix(BASE/c/"matrices"/f"{prefix}_{res//1000}k_KR.{chrom}",mmap=True)
   domains,_=reported(c,chrom);result=dict(cached.get("methods",{}))
   for method,spans in domains.items():
    if cached and method!="deDoc2":continue
    outer=outermost_intervals(sorted(spans));values=clustertad_quality(matrix,outer)
    result[method]=dict(reported_domains=len(spans),outermost_domains=len(outer),
     scored_domains=sum(r["quality"] is not None for r in values),
     mean_quality=mean(r["quality"] for r in values),domains=values)
   write(target,dict(condition=c,chromosome=chrom,methods=result,matrix=str(path.relative_to(ROOT)),seconds=time.monotonic()-start,output_definition_version=2))
   del matrix
   print("ClusterTAD",c,chrom,round(time.monotonic()-start,2),flush=True)


def nipbl(chromosomes):
 base=ROOT/"artifacts/tad_501mel_nipbl"
 for n in chromosomes:
  for fold in [0,1]:
   target=OUT/"nipbl"/f"chr{n}_fold{fold}.json"
   if target.exists():continue
   source=base/f"structural_pilot_chr{n}{'_swapped' if fold else ''}.json"
   data=read(source)
   parents=[(p["start_bin"],p["end_bin"]) for r in data["regions"]
    for p in r["methods_by_source"]["control_train"]["KY-W"]["parent_rows"]]
   outer=outermost_intervals(parents);samples={"control":"GSM7325027" if fold else "GSM7325028",
    "nipbl":"GSM7325029" if fold else "GSM7325030"};scores={}
   for label,accession in samples.items():
    matrix,_=load_dense_matrix(ROOT/"data/processed/501mel_nipbl_10kb"/f"{accession}_10k_KR.chr{n}",mmap=True)
    scores[label]=clustertad_quality(matrix,outer);del matrix
   paired=[]
   for a,b in zip(scores["control"],scores["nipbl"]):
    assert (a["start_bin"],a["end_bin"])==(b["start_bin"],b["end_bin"])
    paired.append(dict(start_bin=a["start_bin"],end_bin=a["end_bin"],control=a["quality"],nipbl=b["quality"],
      delta=b["quality"]-a["quality"] if a["quality"] is not None and b["quality"] is not None else None))
   write(target,dict(chromosome=f"chr{n}",fold=fold,all_selected_parents=len(parents),outermost_parents=len(outer),samples=samples,paired=paired))
   print("NIPBL quality",n,fold,flush=True)


def summarize():
 rows=[];missing=[]
 for c in CONDITIONS:
  for n in range(1,23):
   d=read(OUT/"quality"/c/f"chr{n}.json")
   if not d:missing.append(f"{c}/chr{n}");continue
   rows.extend(dict(condition=c,chromosome=f"chr{n}",method=m,**{k:v for k,v in r.items() if k!="domains"}) for m,r in d["methods"].items())
 summaries=[]
 for c,m in sorted({(r["condition"],r["method"]) for r in rows}):
  subset=[r for r in rows if (r["condition"],r["method"])==(c,m)]
  summaries.append(dict(condition=c,method=m,chromosomes=len(subset),mean_quality=mean(r["mean_quality"] for r in subset),
   scored_domains=sum(r["scored_domains"] for r in subset)))
 paired=[]
 for n in range(1,23):
  for fold in [0,1]:
   d=read(OUT/"nipbl"/f"chr{n}_fold{fold}.json")
   if not d:missing.append(f"nipbl/chr{n}/fold{fold}");continue
   paired.extend(dict(chromosome=f"chr{n}",fold=fold,**r) for r in d["paired"])
 ctcf_data=read(OUT/"ctcf.json")
 write(OUT/"quality_summary.json",dict(complete=not missing,missing=missing,rows=rows,summary=summaries))
 write(OUT/"nipbl_summary.json",dict(complete=len({(r["chromosome"],r["fold"]) for r in paired})==44,
  parents=len(paired),valid_pairs=sum(r["delta"] is not None for r in paired),
  control=mean(r["control"] for r in paired),nipbl=mean(r["nipbl"] for r in paired),delta=mean(r["delta"] for r in paired),
  folds=[dict(fold=f,control=mean(r["control"] for r in paired if r["fold"]==f),
   nipbl=mean(r["nipbl"] for r in paired if r["fold"]==f),delta=mean(r["delta"] for r in paired if r["fold"]==f)) for f in [0,1]],rows=paired))
 csv_out("quality_per_chromosome.csv",rows);csv_out("quality_summary.csv",summaries);csv_out("nipbl_paired_domains.csv",paired)
 write(OUT/"status.json",dict(complete=not missing and ctcf_data.get("complete",False),missing=missing,
  ctcf_complete=ctcf_data.get("complete",False),quality_method_chromosomes=len(rows),
  quality_condition_chromosomes=len({(r["condition"],r["chromosome"]) for r in rows}),nipbl_domain_pairs=len(paired)))
 print(json.dumps(read(OUT/"status.json")),flush=True)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("phase",choices=["ctcf","quality","nipbl","summarize","all"])
 p.add_argument("--conditions",nargs="+",default=list(CONDITIONS),choices=list(CONDITIONS))
 p.add_argument("--chromosomes",nargs="+",type=int,default=list(range(1,23)))
 p.add_argument("--wait-pipeline",action="store_true")
 args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if args.wait_pipeline:
  pipe=ROOT/"artifacts/tad_complete_pipeline_20261024"
  while not (sum(read(pipe/c/"status"/m/f"chr{n}.json").get("status")=="completed" for c in CONDITIONS for m in ["deepTAD","KY-W"] for n in range(1,23))==308
   and sum(read(pipe/c/"robustad/boundary.status.json").get("status")=="completed" for c in CONDITIONS if c.endswith("_5kb"))==3):time.sleep(30)
 if args.phase in ["ctcf","all"]:ctcf()
 if args.phase in ["quality","all"]:quality(args.conditions,args.chromosomes)
 if args.phase in ["nipbl","all"]:nipbl(args.chromosomes)
 if args.phase in ["summarize","all"]:summarize()


if __name__=="__main__":main()
