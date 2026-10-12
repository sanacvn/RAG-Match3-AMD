#!/usr/bin/env python3
"""One persistent worker per configuration; no stale index/config between runs."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--configs", nargs="+", default=["E0", "E1", "E2", "E3", "E4"])
    p.add_argument("--normalizer", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.configs:
        config = ROOT / "configs" / (name+".json")
        if not config.is_file() or "/" in name or ".." in name:
            raise SystemExit("Unknown experiment")
        run = args.output / name
        run.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(RAG_CONFIG=str(config), RAG_SOCKET=f"/tmp/rag-experiment-{os.getpid()}-{name}.sock",
                   RAG_STATE_DIR=str((run/"state").resolve()), RAG_OUTPUT_DIR=str((run/"output").resolve()))
        with (run/"worker.log").open("w") as log:
            worker = subprocess.Popen([sys.executable, str(ROOT/"worker.py")], cwd=ROOT, env=env,
                                      stdout=log, stderr=log, start_new_session=True)
            try:
                # app.py --index waits for readiness, but does not start another
                # worker here: wait through the shared helper's existing socket
                # path after explicit process startup using a bounded polling loop.
                import time
                sys.path.insert(0,str(ROOT))
                from rag.protocol import request
                ready_deadline=time.monotonic()+570
                while time.monotonic()<ready_deadline:
                    if worker.poll() is not None:
                        raise RuntimeError(f"{name}: worker exited; inspect {run}/worker.log")
                    try:
                        request(env["RAG_SOCKET"],{"op":"ping"},.5)
                        break
                    except OSError:
                        time.sleep(.2)
                else:
                    raise TimeoutError(f"{name}: worker startup exceeded budget")
                command = [sys.executable, str(ROOT/"scripts/evaluate.py"), "--index", "--corpus", str(args.corpus.resolve()),
                           "--labels", str(args.labels.resolve()), "--report", str((run/"report.json").resolve())]
                if args.normalizer:
                    command += ["--normalizer", str(args.normalizer.resolve())]
                result=subprocess.run(command, cwd=ROOT, env=env, timeout=1200)
                if result.returncode:
                    raise RuntimeError(f"{name}: evaluation failed")
            finally:
                os.killpg(worker.pid,signal.SIGTERM) if worker.poll() is None else None
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid,signal.SIGKILL)
                    worker.wait()


if __name__ == "__main__":
    main()
