#!/usr/bin/env python3
"""Linux permission test; also run as Docker root with --cap-drop DAC_OVERRIDE."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import Config
from rag.parsers import parse_corpus


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    denied = root / "unreadable.txt"
    denied.write_text("protected value")
    denied.chmod(0)
    (root / "readable.txt").write_text("Visible value: 283")
    cfg = Config(backend="extractive", device="cpu", require_rocm=False)
    evidence, manifest = parse_corpus(root, cfg, lambda *_: "", time.monotonic()+15)
    statuses = {row["path"]:row["status"] for row in manifest}
    if statuses.get("unreadable.txt") != "failed" or statuses.get("readable.txt") != "parsed":
        raise SystemExit("Permission check failed: run non-root or drop DAC_OVERRIDE")
    if [e.source_path for e in evidence] != ["readable.txt"]:
        raise SystemExit("Protected content reached index")
    print(json.dumps({"uid": os.getuid(), "permission_test": "passed", "files": statuses}))
