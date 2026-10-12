"""Persistent model/index owner. No public TCP port."""
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

from rag.config import Config
from rag.pipeline import Pipeline
from rag.protocol import receive, send
from rag.schema import EMPTY


def main():
    config = Config.read()
    socket_path = Path(config.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(str(socket_path)+".lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("A worker already owns this socket")
    pipeline = Pipeline(config)
    try:
        pipeline.restore()
    except Exception as exc:
        print(json.dumps({"event": "restore_failed", "error": str(exc)}), file=sys.stderr, flush=True)
    socket_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    server.listen(8)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(json.dumps({"event": "ready", "pid": os.getpid(), "backend": config.backend,
                      "model_load_seconds": pipeline.model_load_seconds}), flush=True)
    try:
        while True:
            connection, _ = server.accept()
            with connection:
                connection.settimeout(config.startup_timeout)
                try:
                    message = receive(connection)
                    op = message.get("op")
                    if op != "ping" and message.get("config_signature") != config.signature():
                        raise ValueError("Client/worker configuration mismatch; restart worker after configuration changes")
                    if op == "ping":
                        response = {"ok": True, "pid": os.getpid(), "generation": pipeline.generation,
                                    "config_signature": config.signature(),
                                    "indexed_corpus": str(pipeline.root) if pipeline.root else None}
                    elif op == "index":
                        # Deadline provided by the CLI includes its worker-ready wait.
                        seconds = min(config.startup_timeout, float(message["expires_at"])-time.time())
                        stats = pipeline.index(message["corpus"], time.monotonic()+max(0, seconds))
                        response = {"ok": True, "stats": stats}
                    elif op == "query":
                        seconds = min(config.query_timeout-1, float(message["expires_at"])-time.time())
                        if seconds <= 0:
                            raise TimeoutError("Request expired while queued")
                        result, trace = pipeline.query(message["corpus"], message["query"], time.monotonic()+seconds)
                        response = {"ok": True, "output": result, "trace": trace}
                    else:
                        raise ValueError("Unknown worker operation")
                except Exception as exc:
                    response = {"ok": False, "output": dict(EMPTY), "error": type(exc).__name__+": "+str(exc)}
                    print(json.dumps({"event": "request_failed", "error": response["error"]}), file=sys.stderr, flush=True)
                try:
                    send(connection, response)
                except (OSError, ValueError):
                    pass  # Client timed out; it already wrote a valid empty output.
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
