# syntax=docker/dockerfile:1
# Required build argument: obtain the EXACT official image (prefer digest) from
# All About the Challenges. No guessed fallback is supplied.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
WORKDIR /app
COPY requirements.txt scripts/check_torch_build.py /tmp/rag-build/
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi \
 && python3 /tmp/rag-build/check_torch_build.py \
 && python3 -m pip install --no-cache-dir --no-deps -r /tmp/rag-build/requirements.txt \
 && python3 /tmp/rag-build/check_torch_build.py \
 && python3 -m pip check
COPY . /app
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    RAG_REQUIRE_ROCM=true RAG_DEVICE=cuda RAG_OUTPUT_DIR=/app/output \
    RAG_STATE_DIR=/app/state RAG_MODEL_DIR=/app/models RAG_SOCKET=/tmp/match3-rag.sock
RUN test -f /app/models/generator/weights-manifest.json \
 && test -f /app/models/vision/weights-manifest.json
CMD ["bash", "/app/entrypoint.sh"]
