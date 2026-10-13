#!/usr/bin/env python3
"""Copy and re-embed a trusted local evidence snapshot into Chroma Cloud."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chroma_search import CloudStore, cloud_chunks, revision_for
from rag.config import Config
from rag.parsers import fingerprint
from rag.schema import Evidence, atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "state/index.json")
    parser.add_argument("--output", type=Path, help="New Cloud snapshot; never overwrites the input")
    parser.add_argument("--dry-run", action="store_true", help="Validate and count chunks without network access")
    args = parser.parse_args(argv)
    try:
        os.environ["RAG_SEARCH_BACKEND"] = "chroma"
        config = Config.read()
        original = args.snapshot.resolve(strict=True)
        snapshot = json.loads(original.read_text())
        if snapshot.get("schema") not in {1, 2}:
            raise ValueError("Unsupported local snapshot schema")
        corpus = Path(snapshot["corpus"]).resolve(strict=True)
        if original.is_relative_to(corpus):
            raise ValueError("Snapshot must be trusted local state outside the corpus")
        if fingerprint(corpus) != snapshot["fingerprint"]:
            raise ValueError("Corpus changed since the snapshot; reindex locally before migration")
        output = (args.output or Path(config.state_dir) / "chroma-migrated-index.json").resolve()
        if output == original or output.is_relative_to(corpus) or output.exists():
            raise ValueError("Choose a new output snapshot outside the corpus; existing files are preserved")
        chunks = cloud_chunks([Evidence(**e) for e in snapshot["evidence"]], config.chroma_chunk_bytes)
        if args.dry_run:
            print(json.dumps({"chunks": len(chunks), "max_chunk_bytes": max(
                (len(e.content.encode('utf-8')) for e in chunks), default=0), "uploaded": False}))
            return 0
        deadline = time.monotonic() + config.startup_timeout
        identities = [m for m in snapshot["models"] if m["role"] != "embedding"
                      and (config.rerank or m["role"] != "reranker")]
        revision = revision_for(snapshot["fingerprint"], config, identities)
        store = CloudStore(config, corpus, revision, chunks)
        uploaded = store.upload(deadline)
        if fingerprint(corpus) != snapshot["fingerprint"]:
            raise ValueError("Corpus changed during migration; no Cloud snapshot was activated")
        migrated = dict(snapshot, schema=2, config_signature=config.signature(),
                        models=identities, evidence=[e.to_dict() for e in chunks], chroma=store.snapshot())
        atomic_json(output, migrated)
        print(json.dumps({"snapshot": str(output), "uploaded_chunks": uploaded, **store.snapshot()}))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
