from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path=None):
    """A deliberately small dotenv reader: values only, no shell expansion."""
    path = Path(path or os.environ.get("RAG_ENV_FILE", ROOT / ".env"))
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid configuration line in {path}")
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

    @classmethod
    def read(cls):
        load_env()
        config_path = path_value("RAG_CONFIG", ROOT / "configs/E0.json")
        values = json.loads(Path(config_path).read_text())
        allowed = {"method", "rerank", "expand", "chunk_size", "chunk_overlap",
                   "candidates", "context_k", "rrf_c", "expansion_ids"}
        if set(values) - allowed:
            raise ValueError("Unknown experiment configuration keys")
        c = cls(**values)
        c.backend = os.environ.get("RAG_BACKEND", c.backend)
        c.socket = path_value("RAG_SOCKET", c.socket)
        for field in ("state_dir", "output_dir", "model_dir"):
            setattr(c, field, path_value("RAG_" + field.upper(), field.removesuffix("_dir")))
        for role in ("generator", "embedding", "vision", "reranker"):
            setattr(c, role + "_path", path_value("RAG_" + role.upper() + "_PATH", Path(c.model_dir) / role))
        c.device = os.environ.get("RAG_DEVICE", c.device)
        c.require_rocm = boolean("RAG_REQUIRE_ROCM", c.require_rocm)
        c.verify_weights = boolean("RAG_VERIFY_WEIGHTS", c.verify_weights)
        for field in ("query_timeout", "startup_timeout", "parse_timeout"):
            setattr(c, field, float(os.environ.get("RAG_" + field.upper(), getattr(c, field))))
        for field in ("max_file_mb", "max_new_tokens"):
            setattr(c, field, int(os.environ.get("RAG_" + field.upper(), getattr(c, field))))
        if c.backend not in {"transformers", "extractive"} or c.method not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Unknown backend or retrieval method")
        if not 0 <= c.chunk_overlap < c.chunk_size or min(c.candidates, c.context_k, c.rrf_c, c.expansion_ids) < 1:
            raise ValueError("Invalid retrieval/chunk parameters")
        if not 0 < c.query_timeout < 30 or not 0 < c.startup_timeout < 600 or c.parse_timeout <= 0:
            raise ValueError("Deadlines must fit official limits")
        if c.max_file_mb < 1 or c.max_new_tokens < 1:
            raise ValueError("Invalid resource limits")
        if c.backend == "extractive" and (c.method != "bm25" or c.rerank):
            raise ValueError("Extractive diagnostics support BM25 only")
        return c

    def signature(self):
        d = asdict(self)
        for k in ("socket", "output_dir", "query_timeout", "startup_timeout"):
            d.pop(k)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
