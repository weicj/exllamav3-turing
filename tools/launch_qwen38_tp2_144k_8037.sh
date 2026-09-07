#!/usr/bin/env bash
# Start the audited 144K NCCL TP2 service on superserver.
set -euo pipefail

APP_DIR="/home/max/workspace/ExLlamaV3-Turing"
PYTHON_BIN="/home/max/workspace/VLLM-2080ti-0.2.1-pre/.venv/bin/python"
PORT="${QWEN38_PORT:-8037}"
LOG_FILE="$APP_DIR/serve_qwen38_tp2_144k-${PORT}.log"
PID_FILE="$APP_DIR/serve_qwen38_tp2_144k-${PORT}.pid"
SERVER_SCRIPT="tools/serve_qwen38_tp2_144k.py"

GPU0="GPU-1c2b8831-227f-96d1-0c0f-92a189726072"
GPU1="GPU-4da87347-023e-7014-a837-6b1fb649a42e"

if [[ ! -x "$PYTHON_BIN" || ! -f "$APP_DIR/$SERVER_SCRIPT" ]]; then
    echo "Missing target Python environment or server script" >&2
    exit 1
fi

if [[ -s "$PID_FILE" ]]; then
    existing_pid=$(<"$PID_FILE")
    if [[ "$existing_pid" =~ ^[0-9]+$ && -r "/proc/$existing_pid/cmdline" ]] \
        && tr '\0' ' ' < "/proc/$existing_pid/cmdline" | grep -Fq "$SERVER_SCRIPT"; then
        echo "Service is already running as PID $existing_pid" >&2
        exit 1
    fi
fi

if ss -ltnH | awk -v port=":$PORT" '$4 ~ (port "$") { found = 1 } END { exit !found }'; then
    echo "Port $PORT is already listening; refusing to replace an unknown service" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU0,$GPU1"
export NCCL_ALGO="Ring"
export NCCL_PROTO="Simple"
export EXL3_NGRAM_STREAM="1"
export EXL3_FLASHINFER_WORKSPACE_MB="128"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED="1"

cd "$APP_DIR"
if [[ "${1:-}" == "--foreground" ]]; then
    exec "$PYTHON_BIN" -u "$SERVER_SCRIPT" --host 0.0.0.0 --port "$PORT"
fi

if [[ -f "$LOG_FILE" ]]; then
    mv "$LOG_FILE" "$LOG_FILE.$(date -u +%Y%m%dT%H%M%SZ).previous"
fi
nohup setsid "$PYTHON_BIN" -u "$SERVER_SCRIPT" --host 0.0.0.0 --port "$PORT" \
    </dev/null >"$LOG_FILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"
echo "Started PID $pid. Follow $LOG_FILE until /health reports status=ok."
