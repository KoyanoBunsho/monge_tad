"""Remeasure Ours and deepTAD serially after the new accuracy analyses finish."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from tad_runtime_lock import caller_lock

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/tad_no_separator_20261029"
CONDITIONS=("gm12878_25kb","gm12878_10kb","k562_25kb","imr90_25kb",
            "gm12878_5kb","k562_5kb","imr90_5kb")


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data,indent=2)+"\n")
    temporary.replace(path)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    while True:
        missing=[(c,n) for c in CONDITIONS for n in range(1,23)
                 if not read(OUT/"benchmark"/c/f"chr{n}.json").get("complete")]
        biology=read(OUT/"nipbl_summary.json").get("complete",False)
        if not missing and biology:break
        write(OUT/"timing_status.json",dict(status="waiting_for_accuracy",missing_benchmark=len(missing),
                                          biology_complete=biology,updated=time.time()))
        if read(OUT/"status.json").get("status")=="benchmark_queue_finished" and missing:
            raise RuntimeError(f"Benchmark failures require review: {missing}")
        time.sleep(15)
    pipeline=OUT/"pipeline"
    for condition in CONDITIONS:
        for phase in ("boundary","post"):
            write(OUT/"timing_status.json",dict(status="running",condition=condition,phase=phase,updated=time.time()))
            log=OUT/"timing_logs"/f"{condition}.{phase}.log"
            log.parent.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,"-u",str(ROOT/"scripts/benchmark_tad_complete_pipeline.py"),phase,
                     "--conditions",condition,"--output-dir",str(pipeline),"--objective","no_separator",
                     "--reference-root",str(OUT)]
            with caller_lock(),log.open("a") as stream:
                subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            print(condition,phase,"finished",flush=True)
        for method in ("deepTAD","KY-W"):
            for n in range(1,23):
                status=read(pipeline/condition/"status"/method/f"chr{n}.json")
                if status.get("status")!="completed":
                    raise RuntimeError(f"Incomplete timing: {condition} {method} chr{n}: {status}")
    subprocess.run([sys.executable,str(ROOT/"scripts/benchmark_tad_complete_pipeline.py"),"summarize",
                    "--output-dir",str(pipeline),"--objective","no_separator","--reference-root",str(OUT)],
                   cwd=ROOT,check=True)
    write(OUT/"timing_status.json",dict(status="complete",updated=time.time()))


if __name__=="__main__":main()
