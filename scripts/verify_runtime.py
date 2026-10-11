#!/usr/bin/env python3
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import Config
from rag.models import verify_snapshot


def verify():
    import torch
    config = Config.read()
    if config.backend != "transformers" or not config.device.startswith("cuda"):
        raise SystemExit("FAIL: submission requires the real Transformers backend on GPU")
    if not torch.version.hip or not torch.cuda.is_available() or "+rocm" not in torch.__version__:
        raise SystemExit("FAIL: working ROCm torch/GPU required; CPU or CUDA does not pass")
    roles = ["generator", "vision"]
    if config.method != "bm25":
        roles.append("embedding")
    if config.rerank:
        roles.append("reranker")
    identities = [verify_snapshot(getattr(config, role+"_path"), role) for role in roles]
    print(json.dumps({"torch": torch.__version__, "hip": torch.version.hip,
                      "device": torch.cuda.get_device_name(config.device), "models": identities}))


if __name__ == "__main__":
    verify()
