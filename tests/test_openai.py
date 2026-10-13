"""Real OpenAI SDK through a local HTTP boundary, plus existing Chroma SDK."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

try:
    import httpx
    import openai
    from test_chroma import SDKTransport
except ImportError as exc:
    raise unittest.SkipTest("Install requirements-cloud.txt for OpenAI tests") from exc

from rag.config import Config, load_env
from rag.openai_models import OpenAIModels
from rag.pipeline import Pipeline
from rag.schema import Evidence
from scripts.migrate_chroma import main as migrate


class OpenAITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {
            "RAG_ENV_FILE": str(self.root / "none.env"), "RAG_BACKEND": "openai",
            "RAG_SEARCH_BACKEND": "chroma", "RAG_CONFIG": "configs/cloud.json", "CHROMA_SCOPE": "test-user",
            "CHROMA_API_KEY": "chroma-test-only", "OPENAI_API_KEY": "openai-test-only",
            "RAG_STATE_DIR": str(self.root / "state"), "RAG_OUTPUT_DIR": str(self.root / "output"),
            "RAG_MODEL_DIR": str(self.root / "models"), "RAG_DEVICE": "cpu", "RAG_REQUIRE_ROCM": "false"})
        self.env.start()
        self.config = Config.read()
        self.requests = []
        self.status = 200
        self.finish = "stop"
        self.false_quote = False
        self.http = patch("httpx.Client.send", autospec=True, side_effect=self.send)
        self.http.start()
        self.transport = SDKTransport()
        self.embeddings = patch("httpx.Client.post", side_effect=self.transport.post_embeddings)
        self.embeddings.start()
        self.connection = patch("rag.chroma_search.connect", return_value=self.transport)
        self.connection.start()

    def tearDown(self):
        self.connection.stop()
        self.embeddings.stop()
        self.http.stop()
        self.env.stop()
        self.temp.cleanup()

    def send(self, client, request, **kwargs):
        payload = json.loads(request.content)
        self.requests.append(payload)
        if self.status != 200:
            return httpx.Response(self.status, request=request, json={"error": {"message": "unavailable", "type": "server_error"}})
        content = payload["messages"][-1]["content"]
        if isinstance(content, list):
            text = "batch timeout: 235"
        else:
            item = json.loads(content)["evidence"][0]
            text = json.dumps({"answer": "235", "evidence_ids": [item["chunk_id"]],
                "proofs": [{"evidence_id": item["chunk_id"], "quote": "invented quote" if self.false_quote else item["content"],
                            "role": "anchor"}], "links": []})
        return httpx.Response(200, request=request, json={"id": "test", "object": "chat.completion",
            "created": 1, "model": payload["model"], "choices": [{"index": 0, "finish_reason": self.finish,
            "message": {"role": "assistant", "content": text}}]})

    def corpus(self):
        root = self.root / "corpus"
        root.mkdir()
        (root / "record.txt").write_text("batch timeout: 235")
        return root

    def test_config_file_exists_and_openai_is_supported(self):
        self.assertEqual((self.config.backend, self.config.search_backend, self.config.method), ("openai", "chroma", "hybrid"))
        self.assertEqual(self.config.openai_model, "gpt-4.1-mini")
        os.environ["OPENAI_MODEL"] = "another-model"
        self.assertNotEqual(Config.read().signature(), self.config.signature())
        self.assertNotIn("openai-test-only", repr(self.config))

    def test_pipeline_uses_openai_with_chroma_and_restores_without_local_models(self):
        corpus = self.corpus()
        with patch("rag.pipeline.LocalModels", side_effect=AssertionError("No local weights should load")):
            pipeline = Pipeline(self.config)
            pipeline.index(corpus, time.monotonic() + 10)
            result, _ = pipeline.query(corpus, "What is the batch timeout?", time.monotonic() + 10)
            self.assertEqual(result, {"answer": "235", "citations": ["record.txt"]})
            self.assertTrue(Pipeline(self.config).restore())
        self.assertTrue(self.transport.embedding_requests)
        self.assertTrue(self.transport.searches)
        self.assertEqual(self.requests[-1]["response_format"], {"type": "json_object"})

    def test_false_llm_quote_is_rejected(self):
        corpus = self.corpus()
        pipeline = Pipeline(self.config)
        pipeline.index(corpus, time.monotonic() + 10)
        self.false_quote = True
        self.assertEqual(pipeline.query(corpus, "What is the batch timeout?", time.monotonic() + 10)[0],
                         {"answer": "", "citations": []})

    def test_http_failure_has_no_retry_and_incomplete_answers_are_errors(self):
        model = OpenAIModels(self.config)
        evidence = [Evidence("a", "record.txt", "batch timeout: 235", ".txt")]
        self.status = 503
        with self.assertRaises(openai.InternalServerError): model.answer("timeout?", evidence, time.monotonic() + 5)
        self.assertEqual(len(self.requests), 1)
        self.status, self.finish = 200, "length"
        with self.assertRaisesRegex(ValueError, "incomplete"): model.answer("timeout?", evidence, time.monotonic() + 5)

    def test_vision_uses_image_input_and_configured_model(self):
        from PIL import Image
        self.config.openai_vision_model = "vision-model"
        image = self.root / "test.png"
        Image.new("RGB", (20, 20)).save(image)
        self.assertEqual(OpenAIModels(self.config).vision(image, time.monotonic() + 5), "batch timeout: 235")
        self.assertEqual(self.requests[-1]["model"], "vision-model")
        self.assertTrue(self.requests[-1]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_missing_key_and_invalid_config_errors(self):
        os.environ["OPENAI_API_KEY"] = ""
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"): OpenAIModels(self.config)
        os.environ["OPENAI_BASE_URL"] = "http://unsafe.example/v1"
        with self.assertRaisesRegex(ValueError, "HTTPS"): Config.read()
        os.environ["OPENAI_BASE_URL"] = self.config.openai_base_url
        os.environ["RAG_SEARCH_BACKEND"] = "local"
        with self.assertRaisesRegex(ValueError, "requires"): Config.read()

    def test_dotenv_export_and_invalid_line_number_do_not_execute_values(self):
        path = self.root / "test.env"
        path.write_text('export RAG_TEST_CLOUD_VALUE="$(touch /never)"\n')
        load_env(path)
        self.assertEqual(os.environ["RAG_TEST_CLOUD_VALUE"], "$(touch /never)")
        path.write_text('# comment\nOPENAI_API_KEY: secret-not-to-print\n')
        with self.assertRaisesRegex(ValueError, "line 2") as caught: load_env(path)
        self.assertNotIn("secret-not-to-print", str(caught.exception))

    def test_snapshot_migration_updates_llm_identity_and_can_restore(self):
        corpus = self.corpus()
        old = Pipeline(replace(self.config, backend="extractive"))
        old.index(corpus, time.monotonic() + 10)
        source = Path(self.config.state_dir, "index.json")
        output = self.root / "migrated.json"
        self.assertEqual(migrate(["--snapshot", str(source), "--output", str(output)]), 0)
        self.assertEqual(json.loads(output.read_text())["models"][0]["model"], "gpt-4.1-mini")
        source.write_bytes(output.read_bytes())
        self.assertTrue(Pipeline(self.config).restore())


if __name__ == "__main__":
    unittest.main()
