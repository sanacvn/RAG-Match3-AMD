from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def load_env(path=None):
    """A deliberately small dotenv reader: values only, no shell expansion."""
    path = Path(path or os.environ.get("RAG_ENV_FILE", ROOT / ".env"))
    if not path.exists():
        return
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid configuration line {number} in {path}; expected KEY=value")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def path_value(name, default):
    p = Path(os.environ.get(name, str(default))).expanduser()
    return str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())


def boolean(name, default):
    v = os.environ.get(name, str(default)).lower()
    if v not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true or false")
    return v in {"true", "1"}


@dataclass
class Config:
    backend: str = "transformers"
    search_backend: str = "local"
    chroma_host: str = "api.trychroma.com"
    chroma_tenant: str = "11afbef7-38c1-4590-93de-3dc5dc9851f9"
    chroma_database: str = "RAG-MATCH3-AMD"
    chroma_scope: str = ""
    chroma_corpus_id: str = ""
    chroma_batch_size: int = 64
    chroma_chunk_bytes: int = 12288
    chroma_chunks_per_document: int = 3
    method: str = "bm25"
    rerank: bool = False
    expand: bool = False
    chunk_size: int = 384
    chunk_overlap: int = 64
    candidates: int = 20
    context_k: int = 6
    rrf_c: int = 60
    expansion_ids: int = 4
    socket: str = "/tmp/match3-rag.sock"
    state_dir: str = ""
    output_dir: str = ""
    model_dir: str = ""
    generator_path: str = ""
    embedding_path: str = ""
    vision_path: str = ""
    reranker_path: str = ""
    device: str = "cuda"
    require_rocm: bool = True
    verify_weights: bool = True
    query_timeout: float = 26
    startup_timeout: float = 570
    parse_timeout: float = 15
    max_file_mb: int = 100
    max_new_tokens: int = 512
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini"
    openai_vision_model: str = "gpt-4.1-mini"

    @classmethod
    def read(cls):
        load_env()
        backend = os.environ.get("RAG_BACKEND", "transformers")
        config_path = path_value("RAG_CONFIG", ROOT / ("configs/cloud.json" if backend == "openai" else "configs/E0.json"))
        values = json.loads(Path(config_path).read_text())
        allowed = {"method", "rerank", "expand", "chunk_size", "chunk_overlap",
                   "candidates", "context_k", "rrf_c", "expansion_ids"}
        if set(values) - allowed:
            raise ValueError("Unknown experiment configuration keys")
        c = cls(**values)
        c.backend = backend
        c.search_backend = os.environ.get("RAG_SEARCH_BACKEND", "chroma" if backend == "openai" else c.search_backend)
        for field in ("openai_base_url", "openai_model", "openai_vision_model"):
            setattr(c, field, os.environ.get(field.upper(), getattr(c, field)).strip())
        for field, variable in (("chroma_host", "CHROMA_HOST"), ("chroma_tenant", "CHROMA_TENANT"),
                                ("chroma_database", "CHROMA_DATABASE"), ("chroma_scope", "CHROMA_SCOPE"),
                                ("chroma_corpus_id", "RAG_CORPUS_ID")):
            setattr(c, field, os.environ.get(variable, getattr(c, field)))
        for field in ("chroma_batch_size", "chroma_chunk_bytes", "chroma_chunks_per_document"):
            setattr(c, field, int(os.environ.get(field.upper(), getattr(c, field))))
        c.socket = path_value("RAG_SOCKET", c.socket)
        for field in ("state_dir", "output_dir", "model_dir"):
            setattr(c, field, path_value("RAG_" + field.upper(), field.removesuffix("_dir")))
        for role in ("generator", "embedding", "vision", "reranker"):
            setattr(c, role + "_path", path_value("RAG_" + role.upper() + "_PATH", Path(c.model_dir) / role))
        c.device = os.environ.get("RAG_DEVICE", "cpu" if backend == "openai" else c.device)
        c.require_rocm = boolean("RAG_REQUIRE_ROCM", False if backend == "openai" else c.require_rocm)
        c.verify_weights = boolean("RAG_VERIFY_WEIGHTS", c.verify_weights)
        for field in ("query_timeout", "startup_timeout", "parse_timeout"):
            setattr(c, field, float(os.environ.get("RAG_" + field.upper(), getattr(c, field))))
        for field in ("max_file_mb", "max_new_tokens"):
            setattr(c, field, int(os.environ.get("RAG_" + field.upper(), getattr(c, field))))
        if c.backend not in {"transformers", "extractive", "openai"}:
            raise ValueError("RAG_BACKEND must be transformers, extractive, or openai")
        if c.method not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Retrieval method must be bm25, dense, or hybrid")
        if backend == "openai":
            if c.search_backend != "chroma" or c.rerank:
                raise ValueError("OpenAI mode requires RAG_SEARCH_BACKEND=chroma and rerank=false")
            url = urlparse(c.openai_base_url)
            if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("OPENAI_BASE_URL must be an HTTPS API URL without credentials")
            if not c.openai_model or not c.openai_vision_model:
                raise ValueError("OpenAI model names must not be empty")
        if c.search_backend not in {"local", "chroma"}:
            raise ValueError("RAG_SEARCH_BACKEND must be local or chroma")
        if c.search_backend == "chroma":
            if c.chroma_host != "api.trychroma.com":
                raise ValueError("CHROMA_HOST must be api.trychroma.com for Chroma Cloud")
            if not all((c.chroma_tenant.strip(), c.chroma_database.strip(), c.chroma_scope.strip())):
                raise ValueError("Chroma Cloud requires CHROMA_TENANT, CHROMA_DATABASE and CHROMA_SCOPE")
        if not 512 <= c.chroma_chunk_bytes <= 16384 or not 1 <= c.chroma_batch_size <= 256:
            raise ValueError("Chroma chunks must be 512–16384 bytes; batch size must be 1–256")
        if c.chroma_chunks_per_document < 1:
            raise ValueError("CHROMA_CHUNKS_PER_DOCUMENT must be positive")
        if not 0 <= c.chunk_overlap < c.chunk_size or min(c.candidates, c.context_k, c.rrf_c, c.expansion_ids) < 1:
            raise ValueError("Invalid retrieval/chunk parameters")
        if not 0 < c.query_timeout < 30 or not 0 < c.startup_timeout < 600 or c.parse_timeout <= 0:
            raise ValueError("Deadlines must fit official limits")
        if c.max_file_mb < 1 or c.max_new_tokens < 1:
            raise ValueError("Invalid resource limits")
        if c.backend == "extractive" and ((c.search_backend == "local" and c.method != "bm25") or c.rerank):
            raise ValueError("Extractive diagnostics support BM25 only")
        return c

    def signature(self):
        d = asdict(self)
        for k in ("socket", "output_dir", "query_timeout", "startup_timeout"):
            d.pop(k)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
