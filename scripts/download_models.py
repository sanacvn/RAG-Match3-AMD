#!/usr/bin/env python3
"""Provision snapshots before evaluation; freeze actual commit SHAs and file hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.config import Config
from rag.models import verify_snapshot
from rag.schema import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lock", type=Path, default=ROOT / "configs/models.candidates.json")
    p.add_argument("--roles", nargs="+", choices=["generator", "embedding", "vision", "reranker"],
                   default=["generator", "embedding", "vision"])
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--require-pinned", action="store_true", help="reject movable refs for reproducible re-provisioning")
    args = p.parse_args()
    config = Config.read()
    if args.verify_only:
        for role in args.roles:
            print(json.dumps(verify_snapshot(getattr(config, role+"_path"), role)))
        return
    from huggingface_hub import HfApi, snapshot_download
    source = json.loads(args.lock.read_text())
    locked = {"created_at": datetime.now(timezone.utc).isoformat(), "models": {}}
    for role in args.roles:
        model = source["models"][role]
        repo, ref = model["repo_id"], model["revision"]
        if args.require_pinned and not re.fullmatch(r"[0-9a-f]{40}", ref):
            raise ValueError("--require-pinned requires immutable 40-character commit SHAs")
        # Ref is resolved once; the download uses the exact SHA, never 'main'.
        info = HfApi().model_info(repo, revision=ref)
        revision = info.sha
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Model hub returned an invalid revision")
        target = Path(getattr(config, role+"_path"))
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"Snapshot target must be empty: {target}; do not mix revisions")
        snapshot_download(repo_id=repo, revision=revision, local_dir=target,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja",
                                          "*.tiktoken", "LICENSE*", "README.md", "vocab.*", "merges.*"])
        if not list(target.glob("*.safetensors")):
            raise ValueError(f"No safetensors weights downloaded for {role}")
        files = {}
        for path in sorted(target.rglob("*")):
            if not path.is_file() or ".cache" in path.parts:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1024*1024), b""):
                    digest.update(block)
            files[path.relative_to(target).as_posix()] = {"size": path.stat().st_size, "sha256": digest.hexdigest()}
        manifest = {"role": role, "repo_id": repo, "requested_revision": ref,
                    "revision": revision, "created_at": locked["created_at"], "files": files,
                    "model_card": "README.md", "provisioning_policy": "Operator must check official weight/network rules before provisioning"}
        atomic_json(target / "weights-manifest.json", manifest)
        locked["models"][role] = {"repo_id": repo, "revision": revision}
        print(json.dumps({"role": role, "repo_id": repo, "revision": revision, "files": len(files)}))
    atomic_json(Path(config.model_dir) / "models.lock.json", locked)
    print("Archive models/ outside Git and include it in the image before offline evaluation.")


if __name__ == "__main__":
    main()
