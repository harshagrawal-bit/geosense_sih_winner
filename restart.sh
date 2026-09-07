#!/usr/bin/env bash
# Restart the GeoSense server, reliably.
#
# Killing by pidfile is not enough: a start that failed to bind still writes a
# pid, so the file can point at a dead process while an older server keeps the
# port. Always evict whoever actually holds the port, then wait for it to be
# free before binding again.
set -u
cd "$(dirname "$0")"
PORT="${PORT:-8008}"

owner() { ss -lptnH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1; }

PID=$(owner)
if [ -n "${PID:-}" ]; then
  echo "evicting pid $PID from port $PORT"
  kill "$PID" 2>/dev/null
  for _ in $(seq 1 20); do [ -z "$(owner)" ] && break; sleep 0.5; done
  [ -n "$(owner)" ] && { echo "forcing"; kill -9 "$(owner)" 2>/dev/null; sleep 1; }
fi
[ -n "$(owner)" ] && { echo "FAILED: port $PORT still held by $(owner)"; exit 1; }

nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$PORT" \
  > server.log 2>&1 &
echo $! > .server.pid

for _ in $(seq 1 40); do
  curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/api/sources" && break
  sleep 0.5
done
if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/api/sources"; then
  echo "up on http://127.0.0.1:$PORT (pid $(cat .server.pid))"
else
  echo "FAILED to start - last log lines:"; tail -5 server.log; exit 1
fi
