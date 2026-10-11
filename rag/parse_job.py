import json
import contextlib
from pathlib import Path
import sys
from .parsers import parse_native

if __name__ == "__main__":
    try:
        # Some native parsers print status text to stdout. Keep the transport
        # strictly JSON and send diagnostics to stderr instead.
        with contextlib.redirect_stdout(sys.stderr):
            result = parse_native(Path(sys.argv[1]), sys.argv[2])
        print(json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
