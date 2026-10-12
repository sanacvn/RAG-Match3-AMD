#!/usr/bin/env python3
"""One-command local RAG client for a document or corpus directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from rag.config import Config, ROOT
from rag.parsers import SUPPORTED
from rag.schema import validate_output


def prepare_corpus(source, config):
    """Keep the original untouched; the pipeline requires a directory."""
    if source.is_dir():
        return source
    if not source.is_file():
        raise ValueError("Path harus berupa file atau folder dokumen.")
    if source.suffix.lower() not in SUPPORTED:
        raise ValueError("Format file belum didukung: " + source.suffix)
    if source.stat().st_size > config.max_file_mb * 1024 * 1024:
        raise ValueError(f"File melebihi batas {config.max_file_mb} MiB.")
    # A stable private copy allows worker snapshots to remain valid after exit.
    identity = hashlib.sha256(str(source).encode()).hexdigest()
    corpus = Path(config.state_dir) / "ask-inputs" / identity
    corpus.mkdir(parents=True, exist_ok=True)
    target = corpus / source.name
    if corpus.is_symlink() or target.is_symlink():
        raise ValueError("Folder salinan tidak boleh berupa symlink.")
    if any(p != target for p in corpus.iterdir()):
        raise ValueError("Folder salinan berisi file lain; gunakan RAG_STATE_DIR terpisah.")
    # Copy outside the corpus, then replace atomically so incomplete files aren't indexed.
    with tempfile.NamedTemporaryFile(dir=corpus.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return corpus


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tanya RAG tentang satu file atau folder dokumen.")
    parser.add_argument("path", type=Path, help="Path file atau folder, relatif terhadap terminal saat ini")
    parser.add_argument("question", nargs="?", help="Pertanyaan; jika kosong, diminta lewat terminal")
    parser.add_argument("--diagnostic", action="store_true", help="Mode extractive CPU untuk uji teks sederhana")
    args = parser.parse_args(argv)
    try:
        source = args.path.expanduser().resolve(strict=True)
        if not source.is_dir() and not source.is_file():
            raise ValueError("Path harus berupa file atau folder dokumen.")
        question = args.question
        if question is None:
            question = input("Pertanyaan: ")
        question = question.strip()
        if not question:
            raise ValueError("Pertanyaan tidak boleh kosong.")
        if args.diagnostic:
            # Explicit opt-in; don't silently replace real inference with diagnostics.
            os.environ.update(RAG_BACKEND="extractive", RAG_DEVICE="cpu",
                              RAG_REQUIRE_ROCM="false", RAG_CONFIG=str(ROOT / "configs/E0.json"))
        config = Config.read()
        corpus = prepare_corpus(source, config)
        query_id = "ask_" + uuid.uuid4().hex
        cli = [sys.executable, str(ROOT / "app.py")]
        print("Mengindeks dokumen...", file=sys.stderr, flush=True)
        indexed = subprocess.run(cli + ["--index", str(corpus)], cwd=ROOT,
                                 stdout=subprocess.PIPE, text=True)
        if indexed.returncode:
            print(f"Index gagal. Periksa log: {Path(config.state_dir) / 'worker.log'}\n"
                  "Pastikan backend, model dan perangkat sudah dikonfigurasi. "
                  "Restart worker jika konfigurasi berubah.", file=sys.stderr)
            return indexed.returncode
        queried = subprocess.run(cli + ["--corpus", str(corpus), "--query-id", query_id,
                                       "--query", question], cwd=ROOT)
        if queried.returncode:
            return queried.returncode
        output_path = Path(config.output_dir) / (query_id + "_output.json")
        output = validate_output(json.loads(output_path.read_text()))
        print("Jawaban:", output["answer"] or "Tidak ditemukan jawaban dengan bukti yang cukup.")
        if output["citations"]:
            print("Sumber:")
            for citation in output["citations"]:
                original = source / citation if source.is_dir() else source
                print(f"- {original}")
        print(f"JSON: {output_path}")
        return 0
    except (OSError, ValueError, EOFError) as exc:
        print(f"ask: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nDibatalkan.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
