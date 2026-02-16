FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System dependencies ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

# ── Node.js 20 (direct binary — no PPA dependency) ──────────────────
ENV NODE_VERSION=20.18.0
RUN curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz" \
        | tar -xz -C /usr/local --strip-components=1 \
    && node --version && npm --version

# ── Claude Code CLI ──────────────────────────────────────────────────
RUN npm install -g @anthropic-ai/claude-code

# ── Non-root user ────────────────────────────────────────────────────
RUN useradd -m -s /bin/bash trainer \
    && mkdir -p /workspace \
    && chown trainer:trainer /workspace

# ── Python virtual environment ───────────────────────────────────────
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── PyTorch with CUDA 12.4 ──────────────────────────────────────────
# Wheel bundles CUDA runtime; host NVIDIA driver is passed through by
# the NVIDIA Container Toolkit at `docker run --gpus all`.
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124

# ── Project Python dependencies ──────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ── Hand venv ownership to trainer for `pip install -e .` ────────────
RUN chown -R trainer:trainer /opt/venv

# ── Pre-create Claude home so Docker volume inherits ownership ───────
RUN mkdir -p /home/trainer/.claude \
    && chown trainer:trainer /home/trainer/.claude

USER trainer
WORKDIR /workspace

# ── Entrypoint ───────────────────────────────────────────────────────
COPY --chown=trainer:trainer docker/entrypoint.sh /home/trainer/entrypoint.sh
RUN chmod +x /home/trainer/entrypoint.sh

ENTRYPOINT ["/home/trainer/entrypoint.sh"]
CMD ["bash"]
