#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# launch.sh — Start Claude Code in a safe, Docker-isolated environment
#              with --dangerously-skip-permissions for overnight runs.
#
# The container provides:
#   • Full GPU access (RTX 3090 via NVIDIA Container Toolkit)
#   • Python 3.12 + PyTorch CUDA 12.4
#   • Claude Code CLI with your OAuth credentials
#   • Project mounted at /workspace (only writable host path)
#
# Usage:
#   ./docker/launch.sh              # Interactive Claude Code session
#   ./docker/launch.sh bash         # Drop into a shell instead
#   ./docker/launch.sh --build      # Force rebuild before launching
# ─────────────────────────────────────────────────────────────────────
set -e

cd "$(dirname "$0")/.."

BUILD_FLAG=""
CMD_OVERRIDE=""
for arg in "$@"; do
    case "$arg" in
        --build) BUILD_FLAG="--build" ;;
        bash)    CMD_OVERRIDE="bash" ;;
        *)       echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Cambia CFR — Dockerized Claude Code Environment       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  • Container: cambia-cfr                                ║"
echo "║  • GPU:       passthrough via --gpus all                ║"
echo "║  • Writable:  /workspace (bind-mounted project dir)     ║"
echo "║  • Mode:      --dangerously-skip-permissions            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Build if needed (or forced)
if [ -n "$BUILD_FLAG" ] || ! docker image inspect cambia-cfr >/dev/null 2>&1; then
    echo "[launch] Building Docker image..."
    docker compose build
    echo ""
fi

if [ "$CMD_OVERRIDE" = "bash" ]; then
    echo "[launch] Dropping into container shell..."
    docker compose run --rm cfr bash
else
    echo "[launch] Starting Claude Code (dangerously-skip-permissions)..."
    echo "[launch] Use Ctrl+C to stop. Safe to run overnight — container is sandboxed."
    echo ""
    docker compose run --rm cfr claude --dangerously-skip-permissions
fi
