#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    set -- run --rm -i server
fi

# Export so docker compose picks them up for both build args and the user: directive.
export USER_ID="${USER_ID:-$(id -u)}"
export GROUP_ID="${GROUP_ID:-$(id -g)}"

DATA_ROOT="${COLABFIT_DATA_ROOT:-$SCRIPT_DIR/colabfit_data}"
mkdir -p "$DATA_ROOT/models" "$DATA_ROOT/datasets" "$DATA_ROOT/inference_output"

if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    exec docker compose \
        -f "$SCRIPT_DIR/compose.yaml" \
        -f "$SCRIPT_DIR/compose.nvidia.yaml" \
        --project-directory "$SCRIPT_DIR" \
        "$@"
else
    exec docker compose \
        -f "$SCRIPT_DIR/compose.yaml" \
        --project-directory "$SCRIPT_DIR" \
        "$@"
fi
