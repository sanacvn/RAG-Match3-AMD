"""Evaluator CLI: stdlib-only client; models and indexes stay in worker.py."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from rag.config import Config, ROOT
from rag.protocol import request
from rag.schema import EMPTY, atomic_json, validate_output


def ensure_worker(config, deadline):
    try:
        ready = request(config.socket, {"op": "ping"}, min(1, max(.01, deadline-time.monotonic())))
        if ready["config_signature"] != config.signature():
            raise ValueError("Worker configuration changed; restart worker before --index")
        return
    except (OSError, ConnectionError):
        pass
    Path(config.state_dir).mkdir(parents=True, exist_ok=True)
    Path(config.socket).parent.mkdir(parents=True, exist_ok=True)
    # Only indexing may bootstrap a worker for local usage. Container CMD owns
    # the worker in evaluation. flock in worker prevents duplicate model owners.
    proc = None
    # Container CMD may already own the worker while models are loading and no
    # socket exists yet. Wait for that owner instead of spawning a second model.
    with open(config.socket+".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            launch = True
            fcntl.flock(lock, fcntl.LOCK_UN)
        except BlockingIOError:
            launch = False
    if launch:
        with open(Path(config.state_dir, "worker.log"), "ab") as log:
            proc = subprocess.Popen([sys.executable, str(ROOT / "worker.py")], cwd=ROOT,
                                    stdout=log, stderr=log, start_new_session=True)
    while time.monotonic() < deadline:
        try:
            ready = request(config.socket, {"op": "ping"}, min(.5, max(.01, deadline-time.monotonic())))
            if ready["config_signature"] != config.signature():
                raise ValueError("Worker configuration changed; restart worker before --index")
            return
        except (OSError, ConnectionError):
            if proc is not None and proc.poll() is not None:
                raise RuntimeError("Worker exited; inspect state/worker.log")
            time.sleep(.1)
    raise TimeoutError("Worker did not become ready within startup budget")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", metavar="CORPUS")
    parser.add_argument("--corpus")
    parser.add_argument("--query-id")
    parser.add_argument("--query")
    args = parser.parse_args(argv)
    config = Config.read()
    if args.index:
        if args.corpus or args.query_id or args.query is not None:
            parser.error("--index cannot be combined with query options")
        deadline = time.monotonic()+config.startup_timeout
        try:
            ensure_worker(config, deadline)
            remaining = max(.01, deadline-time.monotonic())
            result = request(config.socket, {"op": "index", "corpus": str(Path(args.index).resolve()),
                                            "config_signature": config.signature(),
                                            "expires_at": time.time()+remaining-1}, remaining)
            if not result["ok"]:
                raise RuntimeError(result["error"])
            print(result["stats"])
            return 0
        except Exception as exc:
            print(f"Index failed: {exc}", file=sys.stderr)
            return 1
    if not args.corpus or args.query_id is None or args.query is None:
        parser.error("query requires --corpus, --query-id and --query")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.query_id):
        parser.error("query-id must be a safe filename component")
    start = time.monotonic()
    status, trace = 0, {}
    try:
        # A query NEVER starts a worker, loads model weights, or indexes a corpus.
        response = request(config.socket, {"op": "query", "corpus": str(Path(args.corpus).resolve()),
                                            "config_signature": config.signature(),
                                            "query": args.query, "expires_at": time.time()+config.query_timeout-1},
                           config.query_timeout)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        output = validate_output(response["output"])
        trace = response["trace"]
    except Exception as exc:
        output = dict(EMPTY)
        trace = {"error": type(exc).__name__+": "+str(exc), "reason": "operational_failure"}
        print(trace["error"], file=sys.stderr)
        status = 1
    # Timeout/failure remains a valid output but a nonzero exit and trace prevent
    # the local evaluator from miscounting it as a correct abstention.
    atomic_json(Path(config.output_dir, args.query_id+"_output.json"), output)
    trace.update(query_id=args.query_id, client_seconds=time.monotonic()-start)
    atomic_json(Path(config.state_dir, "traces", args.query_id+".json"), trace)
    return status


if __name__ == "__main__":
    sys.exit(main())
