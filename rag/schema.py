from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import tempfile


@dataclass
class Evidence:
    chunk_id: str
    source_path: str
    content: str
    kind: str
    location: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


EMPTY = {"answer": "", "citations": []}


def validate_output(value):
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
        raise ValueError("answer must be a string")
    citations = value.get("citations")
    if not isinstance(citations, list) or not all(isinstance(x, str) and x for x in citations):
        raise ValueError("citations must be a list of paths")
    if bool(value["answer"]) != bool(citations):
        raise ValueError("Empty answers must have empty citations; answers need evidence")
    return {"answer": value["answer"], "citations": sorted(set(citations))}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, ensure_ascii=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
