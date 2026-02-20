FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates build-essential && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY pyproject.toml README.md LICENSE /app/
COPY src/ /app/src/
WORKDIR /app
RUN pip install --no-cache-dir '.[full]'

RUN python3 -c \
    "from mace.calculators.foundations_models import mace_mp; \
    mace_mp(model='small', return_raw_model=True)"

# ---

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates libgomp1 && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} mcpuser && \
    useradd -u ${USER_ID} -g mcpuser -m -s /bin/bash mcpuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/mace /home/mcpuser/.cache/mace

ENV PATH="/opt/venv/bin:$PATH"

RUN mkdir -p /home/mcpuser/colabfit/models /home/mcpuser/colabfit/datasets /home/mcpuser/colabfit/inference_output && \
    chown -R mcpuser:mcpuser /home/mcpuser && \
    chmod 755 /home/mcpuser && \
    chmod -R a+rX /home/mcpuser/.cache

USER mcpuser
WORKDIR /home/mcpuser

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import colabfit_mcp; print('ok')" || exit 1

ENTRYPOINT ["/opt/venv/bin/colabfit-mcp"]
