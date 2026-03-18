#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    set -- run --rm -i server
fi

# Export so docker compose picks them up for both build args and the user: directive.
# Fallback to 1000 if id is unavailable (e.g. Windows without WSL2).
export USER_ID="${USER_ID:-$(id -u 2>/dev/null || echo 1000)}"
export GROUP_ID="${GROUP_ID:-$(id -g 2>/dev/null || echo 1000)}"

DATA_ROOT="${COLABFIT_DATA_ROOT:-$SCRIPT_DIR/colabfit_data}"
mkdir -p "$DATA_ROOT/models" "$DATA_ROOT/datasets" "$DATA_ROOT/inference_output" "$DATA_ROOT/test_driver_output"
export HOST_DATA_ROOT="$DATA_ROOT"

OS="$(uname -s)"
COMPOSE_FILES="-f $SCRIPT_DIR/compose.yaml"

if [ "$OS" = "Darwin" ]; then
    # macOS: CUDA is never available regardless of nvidia-smi
    COMPOSE_FILES="$COMPOSE_FILES -f $SCRIPT_DIR/compose.cpu.yaml"
elif command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    COMPOSE_FILES="$COMPOSE_FILES -f $SCRIPT_DIR/compose.nvidia.yaml"
fi

exec docker compose $COMPOSE_FILES --project-directory "$SCRIPT_DIR" "$@"
