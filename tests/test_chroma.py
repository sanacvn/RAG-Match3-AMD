"""Exercise the real Chroma SDK with mocked remote transport, not a live Cloud claim."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

try:
    import httpx
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import SearchResult
    from chromadb.types import Collection as CollectionModel
except ImportError as exc:
    raise unittest.SkipTest("Install requirements-chroma.txt to run Chroma integration tests") from exc

from rag.chroma_search import CloudStore, cloud_chunks, split_bytes
from rag.config import Config, ROOT
from rag.pipeline import Pipeline
from rag.schema import Evidence, atomic_json
from scripts.migrate_chroma import main as migrate


class SDKTransport:
    """Minimal server boundary; SDK still validates, embeds and serializes requests."""
    def __init__(self):
        self.collections = {}
        self.records = {}
        self.upserts = 0
        self.searches = []
        self.embedding_requests = []
        self.fail_upload = False

    def post_embeddings(self, url, **kwargs):
        payload = kwargs["json"]
        self.embedding_requests.append((url, payload))
        texts = payload["texts"]
        embeddings = ([{"indices": [1, 10], "values": [1.0, 0.5]} for _ in texts]
                      if str(url).endswith("embed_sparse") else [[1.0, 0.0] for _ in texts])
        return httpx.Response(200, json={"embeddings": embeddings}, request=httpx.Request("POST", url))

    def get_or_create_collection(self, name, schema, embedding_function, metadata):
        if name not in self.collections:
            model = CollectionModel(id=uuid.uuid4(), name=name, configuration_json={},
                                    serialized_schema=schema.serialize_to_json(), metadata=metadata,
                                    dimension=None, tenant="test-tenant", database="test-database")
            self.collections[name] = Collection(client=self, model=model, embedding_function=embedding_function)
            self.records[model.id] = {}
        return self.collections[name]

    def _upsert(self, collection_id, ids, documents, metadatas, embeddings, **kwargs):
        if self.fail_upload:
            raise RuntimeError("simulated upload failure")
        self.upserts += 1
        for cid, doc, metadata, vector in zip(ids, documents, metadatas, embeddings):
            assert len(vector) == 2
            assert metadata["sparse_embedding"].indices
            self.records[collection_id][cid] = (doc, metadata)

    def _get(self, collection_id, ids, **kwargs):
        records = self.records[collection_id]
        found = [cid for cid in ids if cid in records]
        return {"ids": found, "documents": [records[cid][0] for cid in found],
                "metadatas": [records[cid][1] for cid in found], "uris": None}

    def _search(self, collection_id, searches, **kwargs):
        search = searches[0].to_dict()
        self.searches.append(search)
        revision = search["filter"]["revision"]["$eq"]
        found = [(cid, doc, meta) for cid, (doc, meta) in self.records[collection_id].items()
                 if meta["revision"] == revision]
        found = found[:search["limit"]["limit"]]
        return SearchResult(ids=[[r[0] for r in found]], documents=[[r[1] for r in found]],
                            metadatas=[[r[2] for r in found]], scores=[[0.0 for _ in found]], select=[[]])


def cloud_config(temp):
    return Config(backend="extractive", search_backend="chroma", method="hybrid",
                  device="cpu", require_rocm=False, chroma_scope="test-user",
                  state_dir=str(temp / "state"), output_dir=str(temp / "output"),
                  model_dir=str(temp / "models"), socket=str(temp / "worker.sock"))


class ChromaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = cloud_config(self.root)
        self.transport = SDKTransport()
        self.env = patch.dict(os.environ, {"CHROMA_API_KEY": "test-placeholder-not-a-live-key"})
        self.env.start()
        self.http = patch("httpx.Client.post", side_effect=self.transport.post_embeddings)
        self.http.start()
        self.connection = patch("rag.chroma_search.connect", return_value=self.transport)
        self.connection.start()
        self.deadline = time.monotonic() + 60

    def tearDown(self):
        self.connection.stop()
        self.http.stop()
        self.env.stop()
        self.temp.cleanup()

    def evidence(self):
        return [Evidence("one", "document.txt", "batch timeout: 235", ".txt"),
                Evidence("two", "other.txt", "unrelated field: XYZ", ".txt")]

    def store(self, revision="revision-a", config=None, corpus=None):
        return CloudStore(config or self.config, corpus or self.root / "corpus", revision, self.evidence())

    def test_utf8_line_chunking_keeps_all_text_under_limit(self):
        text = "short line\n" + "🌏漢字" * 5000 + "\nlast line\n"
        parts = list(split_bytes(text, 1024))
        self.assertEqual("".join(parts), text)
        self.assertTrue(all(0 < len(p.encode("utf-8")) <= 1024 for p in parts))
        chunks = cloud_chunks([Evidence("long", "long.txt", text, ".txt")], 1024)
        self.assertEqual("".join(e.content for e in chunks), text)
        self.assertEqual(len({e.chunk_id for e in chunks}), len(chunks))
        self.assertTrue(all(e.location["parent_chunk_id"] == "long" for e in chunks))

    def test_sdk_generates_dense_sparse_and_serializes_rrf_groupby(self):
        store = self.store()
        self.assertEqual(store.upload(self.deadline), 2)
        self.assertEqual(store.search("What is the batch timeout?", self.deadline), [0, 1])
        requests = self.transport.embedding_requests
        self.assertTrue(any(str(url).endswith("embed_sparse") for url, _ in requests))
        self.assertTrue(any(payload.get("instructions", "").startswith("Given a question")
                            for _, payload in requests))
        plan = self.transport.searches[-1]
        self.assertEqual(plan["filter"], {"revision": {"$eq": "revision-a"}})
        self.assertEqual(plan["group_by"]["keys"], ["document_id"])
        self.assertEqual(plan["group_by"]["aggregate"]["$min_k"]["k"], 3)
        serialized = json.dumps(plan)
        self.assertIn('"return_rank": true', serialized)
        self.assertIn('"sparse_embedding"', serialized)
        self.assertNotIn("What is the batch timeout?", serialized)  # SDK embedded both query branches.
        schema = store.collection.schema.serialize_to_json()
        dense = schema["keys"]["#embedding"]["float_list"]["vector_index"]["config"]
        self.assertEqual(dense["embedding_function"]["name"], "chroma-cloud-qwen")
        self.assertNotIn("test-placeholder", json.dumps(schema))

    def test_idempotence_revision_isolation_and_sharding(self):
        first = self.store()
        first.upload(self.deadline)
        requests = len(self.transport.embedding_requests)
        self.assertEqual(self.store().upload(self.deadline), 0)
        self.assertEqual(len(self.transport.embedding_requests), requests)
        next_revision = self.store("revision-b")
        next_revision.upload(self.deadline)
        first.search("batch timeout", self.deadline)
        self.assertEqual(self.transport.searches[-1]["filter"]["revision"]["$eq"], "revision-a")
        self.assertNotEqual(first.name, self.store(config=replace(self.config, chroma_scope="other-user")).name)
        self.assertNotEqual(first.name, self.store(corpus=self.root / "another-corpus").name)
        self.assertEqual(first.name, next_revision.name)

    def test_corrupt_cloud_data_and_foreign_collection_rejected(self):
        store = self.store()
        store.upload(self.deadline)
        self.transport.records[store.collection.id][store.ids[0]] = ("tampered", store.metadata[0])
        with self.assertRaisesRegex(ValueError, "modified evidence"):
            store.complete(self.deadline)
        with self.assertRaisesRegex(ValueError, "outside the active"):
            store.search("timeout", self.deadline)
        store.collection._model.metadata = {"integration": "another-app"}
        with self.assertRaisesRegex(ValueError, "ownership"):
            self.store()

    def test_pipeline_restore_query_and_failed_rebuild(self):
        corpus = self.root / "corpus"
        corpus.mkdir()
        (corpus / "data.txt").write_text("batch timeout: 235")
        pipeline = Pipeline(self.config)
        pipeline.index(corpus, self.deadline)
        result, _ = pipeline.query(corpus, "What is the batch timeout?", self.deadline)
        self.assertEqual(result, {"answer": "235", "citations": ["data.txt"]})
        self.assertFalse((self.root / "state/dense.npy").exists())
        restored = Pipeline(self.config)
        self.assertTrue(restored.restore())
        self.assertTrue(restored.index_valid(corpus))
        (corpus / "data.txt").write_text("batch timeout: 260")
        self.assertFalse(restored.index_valid(corpus))
        self.transport.fail_upload = True
        with self.assertRaisesRegex(RuntimeError, "upload failure"):
            restored.index(corpus, self.deadline)
        self.assertIsNone(restored.retriever)
        self.assertFalse((self.root / "state/index.json").exists())

    def test_migration_embeds_snapshot_and_preserves_original(self):
        corpus = self.root / "corpus"
        corpus.mkdir()
        (corpus / "data.txt").write_text("batch timeout: 235")
        local = Pipeline(replace(self.config, search_backend="local", method="bm25"))
        local.index(corpus, self.deadline)
        original = self.root / "state/index.json"
        # Old dense indexes also carried a local embedding model identity.
        snapshot = json.loads(original.read_text())
        snapshot["models"].append({"role": "embedding", "repo_id": "old-bge", "revision": "old"})
        atomic_json(original, snapshot)
        before = original.read_bytes()
        output = self.root / "migrated.json"
        with patch.dict(os.environ, {"CHROMA_SCOPE": "test-user", "RAG_BACKEND": "extractive",
                                    "RAG_DEVICE": "cpu", "RAG_REQUIRE_ROCM": "false",
                                    "RAG_CONFIG": str(ROOT / "configs/chroma.json"),
                                    "RAG_STATE_DIR": str(self.root / "cloud-state"),
                                    "RAG_OUTPUT_DIR": str(self.root / "output"),
                                    "RAG_MODEL_DIR": str(self.root / "models")}):
            self.assertEqual(migrate(["--snapshot", str(original), "--output", str(output)]), 0)
            migrated = json.loads(output.read_text())
            self.assertEqual(migrated["schema"], 2)
            self.assertEqual(migrated["chroma"]["records"], 1)
            self.assertFalse(any(m["role"] == "embedding" for m in migrated["models"]))
            atomic_json(self.root / "cloud-state/index.json", migrated)
            cloud = Pipeline(Config.read())
            self.assertTrue(cloud.restore())
            self.assertEqual(cloud.query(corpus, "What is the batch timeout?", self.deadline)[0]["answer"], "235")
            self.assertEqual(migrate(["--snapshot", str(original), "--output", str(output)]), 1)
        self.assertEqual(original.read_bytes(), before)

    def test_missing_credentials_fail_before_loading_local_models(self):
        with patch.dict(os.environ, {"CHROMA_API_KEY": ""}), patch("rag.pipeline.LocalModels") as models:
            with self.assertRaisesRegex(ValueError, "CHROMA_API_KEY"):
                Pipeline(replace(self.config, backend="transformers"))
            models.assert_not_called()

    def test_unsafe_source_and_expired_deadline_rejected(self):
        store = self.store()
        with self.assertRaises(TimeoutError):
            store.upload(time.monotonic() - 1)
        with self.assertRaisesRegex(ValueError, "relative"):
            CloudStore(self.config, self.root / "corpus", "bad", [Evidence("x", "../escape.txt", "x", ".txt")])


if __name__ == "__main__":
    unittest.main()
