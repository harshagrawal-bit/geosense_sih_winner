#!/usr/bin/env bash
# GeoSense prototype launcher
#harsh agarwal
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { echo "run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8008 "$@"
