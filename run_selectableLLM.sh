#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/src/selectableLLM.py"

if [ ! -f "$PY_SCRIPT" ]; then
  echo "파일이 존재하지 않습니다: $PY_SCRIPT"
  exit 1
fi

python3 "$PY_SCRIPT" "$@"
