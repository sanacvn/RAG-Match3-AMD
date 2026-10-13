"""Chroma Cloud hybrid retrieval; local evidence remains authoritative for citations."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

from .retrieval import Retriever
from .schema import Evidence

SPARSE_KEY = "sparse_embedding"
SCHEMA_VERSION = "match3-qwen-document-search-splade-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Chroma operation exceeded its deadline")


def require_credentials():
    if not os.environ.get("CHROMA_API_KEY", "").strip():
        raise ValueError("Set CHROMA_API_KEY securely in .env or environment settings before using Chroma Cloud")


def split_bytes(text, maximum):
    """Pack whole lines; split an oversized line only at UTF-8 character boundaries."""
    pending = ""
    for line in text.splitlines(keepends=True):
        raw = line.encode("utf-8")
        if len(raw) > maximum:
            if pending:
                yield pending
                pending = ""
            start = 0
            while len(raw) - start > maximum:
                end = start + maximum
                while raw[end] & 0xC0 == 0x80:
                    end -= 1
                yield raw[start:end].decode("utf-8")
                start = end
            line = raw[start:].decode("utf-8")
        if len((pending + line).encode("utf-8")) > maximum:
            yield pending
            pending = ""
        pending += line
    if pending:
        yield pending


def cloud_chunks(evidence, maximum):
    chunks = []
    for e in evidence:
        if len(e.content.encode("utf-8")) <= maximum:
            if e.content.strip():
                chunks.append(e)
            continue
        for number, content in enumerate(split_bytes(e.content, maximum)):
            if not content.strip():
                continue
            cid = digest([e.chunk_id, number, content])[:24]
            location = dict(e.location, parent_chunk_id=e.chunk_id, fragment_index=number)
            chunks.append(Evidence(cid, e.source_path, content, e.kind, location))
    return chunks


def revision_for(fingerprint, config, model_identities):
    return digest([fingerprint, config.signature(), model_identities])


def build_schema():
    from chromadb import K, Schema, SparseVectorIndexConfig, VectorIndexConfig
    from chromadb.utils.embedding_functions import (
        ChromaCloudQwenEmbeddingFunction, ChromaCloudSpladeEmbeddingFunction,
    )
    from chromadb.utils.embedding_functions.chroma_cloud_qwen_embedding_function import (
        ChromaCloudQwenEmbeddingModel, ChromaCloudQwenEmbeddingTarget,
    )
    dense = ChromaCloudQwenEmbeddingFunction(
        model=ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B,
        # SDK 1.5.9 only serializes this task key; override its coding instructions
        # with general document retrieval instructions.
        task="nl_to_code",
        instructions={"nl_to_code": {
            ChromaCloudQwenEmbeddingTarget.DOCUMENTS: "",
            ChromaCloudQwenEmbeddingTarget.QUERY:
                "Given a question, retrieve relevant document passages that answer the question.",
        }},
    )
    sparse = ChromaCloudSpladeEmbeddingFunction()
    dense.validate_config(dense.get_config())
    sparse.validate_config(sparse.get_config())
    schema = Schema().create_index(config=VectorIndexConfig(
        space="cosine", source_key=K.DOCUMENT, embedding_function=dense))
    schema.create_index(key=SPARSE_KEY, config=SparseVectorIndexConfig(
        source_key=K.DOCUMENT, embedding_function=sparse))
    return schema, dense


def connect(config):
    require_credentials()
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise RuntimeError("Install requirements-chroma.txt with the Python used by the worker") from exc
    return chromadb.CloudClient(
        tenant=config.chroma_tenant, database=config.chroma_database,
        api_key=os.environ["CHROMA_API_KEY"], settings=Settings(anonymized_telemetry=False))


class CloudStore:
    def __init__(self, config, corpus, revision, evidence, client=None):
        self.config, self.revision, self.evidence = config, revision, evidence
        scope = digest(config.chroma_scope)
        corpus_id = digest(config.chroma_corpus_id or str(Path(corpus).resolve()))
        schema_id = digest([SCHEMA_VERSION, config.chroma_chunk_bytes])
        self.name = f"match3-{scope[:16]}-{corpus_id[:16]}-{schema_id[:12]}"
        self.ids = [digest([revision, e.chunk_id]) for e in evidence]
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate evidence IDs in corpus")
        indices = defaultdict(int)
        self.metadata = []
        for e in evidence:
            source = Path(e.source_path)
            if not source.parts or source.is_absolute() or ".." in source.parts:
                raise ValueError("Evidence source path must be relative to its corpus")
            if len(e.content.encode("utf-8")) > 16384:
                raise ValueError("Chroma documents cannot exceed 16 KiB")
            self.metadata.append({"revision": revision,
                                  "document_id": digest([scope, corpus_id, e.source_path]),
                                  "source_path": e.source_path, "chunk_index": indices[e.source_path],
                                  "chunk_id": e.chunk_id, "kind": e.kind,
                                  "location": json.dumps(e.location, sort_keys=True)})
            indices[e.source_path] += 1
        client = client if client is not None else connect(config)
        schema, _ = build_schema()
        ownership = {"integration": SCHEMA_VERSION, "scope_hash": scope,
                     "corpus_hash": corpus_id, "schema_hash": schema_id}
        # The dense and sparse embedding functions are already bound in Schema.
        # chromadb 1.5.x converts a non-None embedding_function argument into the
        # legacy collection configuration, and Chroma Cloud rejects requests that
        # set both collection configuration and schema.
        self.collection = client.get_or_create_collection(
            name=self.name, schema=schema, embedding_function=None, metadata=ownership)
        # get_or_create ignores schema/metadata on existing collections. Fail closed.
        if any((self.collection.metadata or {}).get(k) != v for k, v in ownership.items()):
            raise ValueError("Chroma collection ownership/schema mismatch; use a distinct scope or corpus ID")
        actual = self.collection.schema.serialize_to_json()
        expected = schema.serialize_to_json()
        def binding(serialized, key):
            entry = serialized.get("keys", {}).get(key, {})
            if key == "#embedding":
                entry = entry.get("float_list", {}).get("vector_index", {})
            else:
                entry = entry.get("sparse_vector", {}).get("sparse_vector_index", {})
            config = entry.get("config", {})
            # Cloud may add algorithm defaults (SPANN/HNSW). Compare semantic
            # bindings rather than rejecting harmless server-supplied defaults.
            return {"enabled": entry.get("enabled"), "source_key": config.get("source_key"),
                    "embedding_function": config.get("embedding_function"), "space": config.get("space")}
        for key in ("#embedding", SPARSE_KEY):
            if binding(actual, key) != binding(expected, key):
                raise ValueError("Chroma collection embedding schema differs from this application")
        self.positions = {cid: i for i, cid in enumerate(self.ids)}

    def existing(self, offset, end, deadline):
        check_time(deadline)
        rows = self.collection.get(ids=self.ids[offset:end], include=["documents", "metadatas"])
        check_time(deadline)
        present = set()
        for cid, document, metadata in zip(rows["ids"], rows["documents"], rows["metadatas"]):
            position = self.positions.get(cid)
            if position is None or document != self.evidence[position].content:
                raise ValueError("Chroma contains unexpected or modified evidence")
            if any(metadata.get(k) != v for k, v in self.metadata[position].items()):
                raise ValueError("Chroma evidence metadata mismatch")
            present.add(cid)
        return present

    def upload(self, deadline):
        uploaded = 0
        size = self.config.chroma_batch_size
        for offset in range(0, len(self.ids), size):
            end = min(offset + size, len(self.ids))
            existing = self.existing(offset, end, deadline)
            missing = [i for i in range(offset, end) if self.ids[i] not in existing]
            if not missing:
                continue
            check_time(deadline)
            # SDK generates Qwen dense + SPLADE sparse embeddings from documents.
            self.collection.upsert(ids=[self.ids[i] for i in missing],
                                   documents=[self.evidence[i].content for i in missing],
                                   metadatas=[self.metadata[i] for i in missing])
            check_time(deadline)
            if len(self.existing(offset, end, deadline)) != end - offset:
                raise RuntimeError("Chroma upload incomplete; retry indexing before querying")
            uploaded += len(missing)
        return uploaded

    def complete(self, deadline):
        size = self.config.chroma_batch_size
        return all(len(self.existing(i, min(i + size, len(self.ids)), deadline)) == len(self.ids[i:i+size])
                   for i in range(0, len(self.ids), size))

    def search(self, query, deadline):
        from chromadb import K, Knn, Rrf, Search
        from chromadb.execution.expression.operator import GroupBy, MinK
        check_time(deadline)
        c = self.config
        if not self.ids:
            return []
        dense = Knn(query=query, key=K.EMBEDDING, limit=c.candidates,
                    return_rank=True, default=c.candidates * 10)
        sparse = Knn(query=query, key=SPARSE_KEY, limit=c.candidates,
                     return_rank=True, default=c.candidates * 10)
        rank = Rrf(ranks=[dense, sparse], weights=[0.7, 0.3], k=c.rrf_c)
        if c.method == "dense":
            rank = Knn(query=query, key=K.EMBEDDING, limit=c.candidates)
        elif c.method == "bm25":
            # In Cloud this selects SPLADE lexical search, not local BM25 scores.
            rank = Knn(query=query, key=SPARSE_KEY, limit=c.candidates)
        search = (Search().where(K("revision") == self.revision).rank(rank)
                  .group_by(GroupBy(keys=K("document_id"),
                                    aggregate=MinK(keys=K.SCORE, k=c.chroma_chunks_per_document)))
                  .limit(c.candidates).select(K.DOCUMENT, K.SCORE, "revision", "chunk_id", "source_path"))
        rows = self.collection.search(search).rows()[0]
        check_time(deadline)
        positions = []
        for row in rows:
            i = self.positions.get(row["id"])
            if i is None or row.get("document") != self.evidence[i].content:
                raise ValueError("Search returned evidence outside the active corpus revision")
            metadata = row.get("metadata") or {}
            for key in ("revision", "chunk_id", "source_path"):
                if metadata.get(key) != self.metadata[i][key]:
                    raise ValueError("Search returned mismatched evidence metadata")
            if i not in positions:
                positions.append(i)
        return positions

    def snapshot(self):
        return {"collection": self.name, "revision": self.revision, "records": len(self.ids)}


class CloudRetriever(Retriever):
    def __init__(self, evidence, config, models, corpus, revision, deadline):
        self.evidence, self.config, self.models = evidence, config, models
        self.vectors = None
        self.deadline = deadline
        self.store = CloudStore(config, corpus, revision, evidence)
        self.uploaded = 0

    def build_dense(self):
        self.uploaded = self.store.upload(self.deadline)

    def rank(self, query, deadline):
        return self.store.search(query, deadline)
