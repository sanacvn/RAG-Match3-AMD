import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest

from rag.config import ROOT
from rag.protocol import request


class AskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ask-test-")
        self.root = Path(self.temp.name)
        self.socket = self.root / "worker.sock"
        self.env = dict(os.environ, RAG_ENV_FILE=str(self.root / "unused.env"),
                        RAG_SOCKET=str(self.socket), RAG_STATE_DIR=str(self.root / "state"),
                        RAG_OUTPUT_DIR=str(self.root / "output"), RAG_MODEL_DIR=str(self.root / "models"))

    def tearDown(self):
        try:
            pid = request(str(self.socket), {"op": "ping"}, 1)["pid"]
            os.kill(pid, signal.SIGTERM)
        except (OSError, ConnectionError):
            pass
        self.temp.cleanup()

    def ask(self, path, question=None, input=None):
        command = [str(ROOT / "ask"), "--diagnostic", str(path)]
        if question is not None:
            command.append(question)
        return subprocess.run(command, cwd=self.root, env=self.env, input=input,
                              text=True, capture_output=True, timeout=40)

    def outputs(self):
        return [json.loads(p.read_text()) for p in (self.root / "output").glob("*_output.json")]

    def test_single_file_and_changed_contents(self):
        source = self.root / "my document.txt"
        for value in ("235", "260"):
            source.write_text(f"batch timeout: {value}\n")
            result = self.ask(source.name, "What is the batch timeout?")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"Jawaban: {value}", result.stdout)
            self.assertIn(str(source), result.stdout)
            self.assertEqual(source.read_text(), f"batch timeout: {value}\n")
        self.assertEqual({p["answer"] for p in self.outputs()}, {"235", "260"})
        self.assertTrue(all(p["citations"] == [source.name] for p in self.outputs()))

    def test_folder_and_interactive_question(self):
        corpus = self.root / "documents"
        corpus.mkdir()
        (corpus / "record.txt").write_text("batch timeout: 235\n")
        result = self.ask(corpus, input="What is the batch timeout?\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Pertanyaan:", result.stdout)
        self.assertEqual(self.outputs(), [{"answer": "235", "citations": ["record.txt"]}])

    def test_unanswerable_question_is_not_an_operational_failure(self):
        source = self.root / "record.txt"
        source.write_text("batch timeout: 235\n")
        result = self.ask(source, "What is the unit price?")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Tidak ditemukan jawaban", result.stdout)
        self.assertEqual(self.outputs(), [{"answer": "", "citations": []}])

    def test_invalid_input_does_not_start_worker(self):
        result = self.ask("missing.txt", "What is the price?")
        self.assertNotEqual(result.returncode, 0)
        source = self.root / "record.txt"
        source.write_text("price: 235\n")
        result = self.ask(source, "  ")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Pertanyaan tidak boleh kosong", result.stderr)
        result = self.ask(source, input="")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.socket.exists())


if __name__ == "__main__":
    unittest.main()
