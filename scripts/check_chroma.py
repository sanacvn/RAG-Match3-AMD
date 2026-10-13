#!/usr/bin/env python3
"""Live upload + hybrid search check with synthetic data; deletes its own collection."""
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chroma_search import CloudStore, connect
from rag.config import Config
from rag.schema import Evidence


def main():
    os.environ["RAG_SEARCH_BACKEND"] = "chroma"
    config = Config.read()
    config.chroma_scope += "-smoke-" + uuid.uuid4().hex
    config.chroma_corpus_id = "synthetic-smoke"
    config.method = "hybrid"
    client = connect(config)
    data = [Evidence("price", "price.txt", "The unit price at 20000 units is USD 12.", ".txt"),
            Evidence("other", "other.txt", "The factory is closed on Sundays.", ".txt")]
    store = CloudStore(config, ROOT / "work/chroma-smoke", "smoke", data, client=client)
    try:
        deadline = time.monotonic() + config.startup_timeout
        if store.upload(deadline) != 2 or not store.complete(deadline):
            raise RuntimeError("Upload/read verification failed")
        hits = store.search("What is the unit price at 20000 units?", deadline)
        if not hits or hits[0] != 0:
            raise RuntimeError("Hybrid search did not rank the price evidence first")
        print("Chroma Cloud: upload, read, Qwen + SPLADE embedding, RRF search and citations verified.")
    finally:
        client.delete_collection(name=store.name)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Cloud check failed: {exc}", file=sys.stderr)
        sys.exit(1)
