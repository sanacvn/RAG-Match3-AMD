#!/usr/bin/env python3
"""Build-time check does not need a GPU; runtime preflight does."""
import torch

if not torch.version.hip or "+rocm" not in torch.__version__:
    raise SystemExit(f"ROCm build was missing/replaced: {torch.__version__}")
print(torch.__version__)
