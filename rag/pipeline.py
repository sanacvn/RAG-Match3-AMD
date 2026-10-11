from dataclasses import asdict
import json
from pathlib import Path
import time

from .evidence import select_output
from .models import ExtractiveDiagnostics, LocalModels
from .parsers import fingerprint, parse_corpus
from .retrieval import Retriever
from .schema import Evidence, atomic_json


class Pipeline:
    def __init__(self, config):
        self.started = time.monotonic()
        self.config = config
        self.models = LocalModels(config) if config.backend == "transformers" else ExtractiveDiagnostics()
        self.retriever = None
        self.root = self.corpus_fingerprint = None
        self.generation = 0
        self.indexed_this_boot = False
        self.state = Path(config.state_dir)
        self.state.mkdir(parents=True, exist_ok=True)
        self.model_load_seconds = time.monotonic()-self.started
        if self.model_load_seconds >= config.startup_timeout:
            raise TimeoutError("Model loading exceeded startup budget")

    def _root(self, corpus):
        root = Path(corpus).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Corpus must be a directory")
        # No generated output, index or model weights may leak back into evidence.
        for path in (self.config.state_dir, self.config.output_dir, self.config.model_dir):
            if Path(path).resolve().is_relative_to(root):
                raise ValueError("State, output and model directories must be outside corpus")
        return root

    def index(self, corpus, deadline):
        # An explicit --index ALWAYS rebuilds. Failed rebuild cannot leave old
        # evidence available, even when the corpus path is the same.
        self.retriever = None
        self.root = self.corpus_fingerprint = None
        self.state.joinpath("index.json").unlink(missing_ok=True)
        self.state.joinpath("dense.npy").unlink(missing_ok=True)
        start = time.monotonic()
        if not self.indexed_this_boot:
            deadline = min(deadline, self.started+self.config.startup_timeout)
        root = self._root(corpus)
        before = fingerprint(root)
        evidence, manifest = parse_corpus(root, self.config, self.models.vision, deadline)
        atomic_json(self.state / "manifest.json", {"corpus": str(root), "files": manifest})
        retriever = Retriever(evidence, self.config, self.models)
        retriever.build_dense()
        if time.monotonic() >= deadline:
            raise TimeoutError("Indexing exceeded startup budget; index is unavailable")
        after = fingerprint(root)
        if before != after:
            raise ValueError("Corpus changed during indexing; run --index again on a stable corpus")
        if retriever.vectors is not None:
            import numpy as np
            np.save(self.state / "dense.npy", retriever.vectors, allow_pickle=False)
        self.generation += 1
        snapshot = {"schema": 1, "corpus": str(root), "fingerprint": after,
                    "config_signature": self.config.signature(), "generation": self.generation,
                    "models": self.models.identities, "evidence": [e.to_dict() for e in evidence]}
        atomic_json(self.state / "index.json", snapshot)
        self.retriever, self.root, self.corpus_fingerprint = retriever, root, after
        self.indexed_this_boot = True
        stats = {"chunks": len(evidence), "files": len(manifest), "index_seconds": time.monotonic()-start,
                 "model_load_seconds": self.model_load_seconds,
                 "startup_elapsed_seconds": time.monotonic()-self.started,
                 "generation": self.generation, **self.models.metrics()}
        atomic_json(self.state / "startup.json", stats)
        return stats

    def restore(self):
        """Restore only on worker boot. Queries never load models or build an index."""
        path = self.state / "index.json"
        if not path.exists():
            return False
        snapshot = json.loads(path.read_text())
        root = self._root(snapshot["corpus"])
        if (snapshot["schema"] != 1 or snapshot["config_signature"] != self.config.signature()
                or snapshot["models"] != self.models.identities or snapshot["fingerprint"] != fingerprint(root)):
            return False
        vectors = None
        if self.config.method != "bm25":
            import numpy as np
            vectors = np.load(self.state / "dense.npy", allow_pickle=False)
            if vectors.shape[0] != len(snapshot["evidence"]):
                return False
        self.retriever = Retriever([Evidence(**e) for e in snapshot["evidence"]], self.config, self.models, vectors)
        self.root, self.corpus_fingerprint = root, snapshot["fingerprint"]
        self.generation = snapshot["generation"]
        return True

    def query(self, corpus, query, deadline):
        start = time.monotonic()
        root = self._root(corpus)
        if self.retriever is None or root != self.root:
            raise ValueError("Corpus has no active index; invoke --index for this corpus first")
        if fingerprint(root) != self.corpus_fingerprint:
            raise ValueError("Corpus changed since indexing; invoke --index again")
        candidates, retrieval = self.retriever.search(query, deadline)
        retrieved_at = time.monotonic()
        internal = self.models.answer(query, candidates, deadline) if candidates else {"answer": ""}
        result, reason = select_output(internal, candidates, query)
        if time.monotonic() >= deadline:
            raise TimeoutError("Query budget exhausted")
        trace = {"query": query, "generation": self.generation, "corpus": str(root),
                 "retrieval": retrieval, "candidate_ids": [e.chunk_id for e in candidates],
                 "candidate_sources": sorted({e.source_path for e in candidates}),
                 "selected_evidence_ids": internal.get("evidence_ids", []), "reason": reason,
                 "retrieval_seconds": retrieved_at-start,
                 "generation_validation_seconds": time.monotonic()-retrieved_at,
                 "seconds": time.monotonic()-start, **self.models.metrics()}
        return result, trace
