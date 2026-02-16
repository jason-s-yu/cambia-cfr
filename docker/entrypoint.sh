#!/bin/bash
set -e

# ── Claude Code authentication ───────────────────────────────────────
# Credentials and settings are bind-mounted read-only under /auth.
# Copy them into the writable Claude home so the CLI can function.
mkdir -p "$HOME/.claude"

if [ -f /auth/.credentials.json ]; then
    cp /auth/.credentials.json "$HOME/.claude/.credentials.json"
    chmod 600 "$HOME/.claude/.credentials.json"
    echo "[entrypoint] Claude credentials loaded."
else
    echo "[entrypoint] WARNING: No credentials found at /auth/.credentials.json"
    echo "             Mount with:  -v \$HOME/.claude/.credentials.json:/auth/.credentials.json:ro"
fi

if [ -f /auth/settings.json ]; then
    cp /auth/settings.json "$HOME/.claude/settings.json"
    echo "[entrypoint] Claude global settings loaded."
fi

# ── Install project in editable mode ─────────────────────────────────
# Only runs once per container start; the source is bind-mounted so
# code changes are reflected immediately.
if [ -f /workspace/pyproject.toml ]; then
    pip install -e /workspace --quiet 2>/dev/null || {
        echo "[entrypoint] WARNING: pip install -e failed; CLI entry point may not work."
        echo "             You can still run:  python -m src.cli ..."
    }
    echo "[entrypoint] Project installed in editable mode."
fi

# ── GPU check ─────────────────────────────────────────────────────────
python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
    print(f'[entrypoint] GPU: {name} ({mem:.1f} GB VRAM)')
else:
    print('[entrypoint] No GPU detected — training will use CPU.')
" || echo "[entrypoint] GPU check skipped (non-critical)."

echo "[entrypoint] Ready. Executing: $*"
echo ""

exec "$@"
