#!/usr/bin/env python3
"""Separate CLI processes per query; labels never enter worker or index."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.config import Config
from rag.schema import atomic_json, validate_output


def normalize(answer):
    # Local approximation only: official self-check is authoritative and missing.
    return re.sub(r"[\s\-.·_]", "", answer.upper())


def canonical_citations(paths, corpus):
    result = set()
    for value in paths:
        p = Path(value)
        if p.is_absolute():
            p = p.resolve().relative_to(corpus.resolve())
        if ".." in p.parts:
            raise ValueError("Unsafe citation")
        result.add(p.as_posix())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--index", action="store_true")
    p.add_argument("--normalizer", type=Path, help="Trusted adapter exporting normalize(answer); never load from corpus")
    args = p.parse_args()
    corpus = args.corpus.resolve()
    if args.labels.resolve().is_relative_to(corpus) or args.report.resolve().is_relative_to(corpus):
        raise SystemExit("Labels and report must be outside corpus")
    normalizer = normalize
    if args.normalizer:
        if args.normalizer.resolve().is_relative_to(corpus):
            raise SystemExit("Normalizer must be outside untrusted corpus")
        spec = importlib.util.spec_from_file_location("trusted_normalizer", args.normalizer)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        normalizer = module.normalize
    config = Config.read()
    startup = None
    if args.index:
        start = time.monotonic()
        proc = subprocess.run([sys.executable, str(ROOT/"app.py"), "--index", str(corpus)], timeout=600)
        startup = {"client_index_seconds": time.monotonic()-start, "exit_code": proc.returncode}
        if proc.returncode:
            raise SystemExit("Index failed; see manifest and worker.log")
    rows = []
    manifest_path = Path(config.state_dir, "manifest.json")
    file_status = {row["path"]: row for row in json.loads(manifest_path.read_text())["files"]} if manifest_path.exists() else {}
    for case in json.loads(args.labels.read_text()):
        qid = case["query_id"]
        out = Path(config.output_dir, qid+"_output.json")
        tracepath = Path(config.state_dir, "traces", qid+".json")
        out.unlink(missing_ok=True)
        tracepath.unlink(missing_ok=True)
        start = time.monotonic()
        failure = None
        try:
            proc = subprocess.run([sys.executable, str(ROOT/"app.py"), "--corpus", str(corpus),
                                   "--query-id", qid, "--query", case["query"]], timeout=30, capture_output=True, text=True)
            predicted = validate_output(json.loads(out.read_text()))
            if proc.returncode:
                failure = proc.stderr.strip() or "CLI failure"
        except Exception as exc:
            failure = type(exc).__name__+": "+str(exc)
            predicted = {"answer": "", "citations": []}
        seconds = time.monotonic()-start
        trace = json.loads(tracepath.read_text()) if tracepath.exists() else {}
        gold = canonical_citations(case["expected_citations"], corpus)
        answer_ok = not failure and normalizer(predicted["answer"]) == normalizer(case["expected_answer"])
        try:
            citation_ok = not failure and canonical_citations(predicted["citations"], corpus) == gold
        except ValueError:
            citation_ok = False
        retrieved = set(trace.get("candidate_sources", []))
        recall = len(gold & retrieved)/len(gold) if gold else None
        error_category = None
        if failure:
            error_category = "timeout" if "timeout" in failure.lower() or "expired" in failure.lower() else "operational"
        elif not (answer_ok and citation_ok):
            if gold-retrieved:
                missing_files = [file_status.get(source, {}) for source in gold-retrieved]
                error_category = "parsing" if any(row.get("status") in {"failed", "skipped", "partial"} for row in missing_files) else "retrieval"
            elif trace.get("reason") in {"incomplete_chain", "unsupported_link", "invalid_link"}:
                error_category = "chain"
            elif case.get("category") in {"revision", "historical_revision"} and predicted["answer"]:
                error_category = "revision"
            elif not predicted["answer"] or not case["answerable"]:
                error_category = "abstention"
            elif not answer_ok:
                error_category = "extraction"
            else:
                error_category = "citation"
        rows.append({**case, "predicted": predicted, "answer_correct": answer_ok,
                     "citations_correct": citation_ok, "joint_correct": answer_ok and citation_ok,
                     "source_recall": recall, "complete_evidence_hit": gold <= retrieved if gold else None,
                     "seconds": seconds, "operational_error": failure, "error_category": error_category, "trace": trace})
    n = len(rows)
    times = sorted(r["seconds"] for r in rows)
    answerable = [r for r in rows if r["answerable"]]
    missing = [r for r in rows if not r["answerable"]]
    def avg(values):
        return statistics.mean(values) if values else None
    report = {"backend": config.backend, "experiment": config.method,
              "rerank": config.rerank, "expand": config.expand,
              "normalization": "official adapter" if args.normalizer else "local approximation; not official scoring",
              "startup": startup, "questions": n,
              "answer_accuracy": avg([r["answer_correct"] for r in rows]),
              "citation_exact_set_accuracy": avg([r["citations_correct"] for r in rows]),
              "joint_accuracy": avg([r["joint_correct"] for r in rows]),
              "source_recall_at_context_k": avg([r["source_recall"] for r in answerable]),
              "complete_evidence_hit_at_context_k": avg([r["complete_evidence_hit"] for r in answerable]),
              "correct_rejection_rate": avg([not r["predicted"]["answer"] and not r["operational_error"] for r in missing]),
              "false_rejection_rate": avg([not r["predicted"]["answer"] and not r["operational_error"] for r in answerable]),
              "false_answer_count": sum(bool(r["predicted"]["answer"]) for r in missing),
              "operational_failures": sum(bool(r["operational_error"]) for r in rows),
              "failure_categories": dict(Counter(r["error_category"] for r in rows if r["error_category"])),
              "max_query_seconds": max(times, default=0),
              "p95_query_seconds": times[max(0, int(.95*len(times)+.999)-1)] if times else None,
              "total_query_seconds": sum(times), "rows": rows}
    atomic_json(args.report, report)
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
