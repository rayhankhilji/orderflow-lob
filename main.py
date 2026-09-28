"""Vercel entrypoint — detected automatically by the Python runtime.

Vercel routes all traffic to this function; paths arrive untouched, so the
FastAPI app's /api/* routes match directly. Static frontend lives in public/
(built by vercel.json's buildCommand) and is served at the edge.

ORDERFLOW_SYNC=1 switches the API into serverless mode: POST endpoints run
inline and return results immediately (no background jobs), with request
sizes clamped to fit inside one function invocation.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ORDERFLOW_SYNC", "1")

from orderflow.api.app import app

# public/ is edge-served when outputDirectory is honored; mount it inside the
# function too so "/" never 404s regardless of routing order.
_PUBLIC = Path(__file__).resolve().parent / "public"
if _PUBLIC.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_PUBLIC), html=True), name="web")

__all__ = ["app"]
