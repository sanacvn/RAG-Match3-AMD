#!/usr/bin/env bash
set -euo pipefail
cd /app
# Worker stays alive and holds all models/index. No HTTP service or public port.
exec python3 -u /app/worker.py
