FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV KIM_API_VERSION=2.4.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake gfortran wget ca-certificates pkg-config \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    python3 -m ensurepip --upgrade && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q "https://s3.openkim.org/kim-api/kim-api-${KIM_API_VERSION}.txz" \
    -O /tmp/kim-api.txz && \
    echo "225e3136d43e416a4424551e9e5f6d92cc6ecfe11389a1b6e97d6dcdfed83d44  /tmp/kim-api.txz" | sha256sum -c - && \
    tar -xJf /tmp/kim-api.txz -C /tmp && \
    rm /tmp/kim-api.txz && \
    cd /tmp/kim-api-${KIM_API_VERSION} && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j"$(nproc)" && make install && \
    ldconfig && \
    rm -rf /tmp/kim-api-${KIM_API_VERSION}

COPY pyproject.toml README.md LICENSE /app/
COPY src/ /app/src/
WORKDIR /app
RUN python3 -m pip install --no-cache-dir '.[full]'

RUN python3 -c \
    "from mace.calculators.foundations_models import mace_mp; \
    mace_mp(model='small', return_raw_model=True)"

# ---

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates libgomp1 libgfortran5 wget && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    rm -rf /var/lib/apt/lists/*

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} mcpuser && \
    useradd -u ${USER_ID} -g mcpuser -m -s /bin/bash mcpuser

COPY --from=builder /usr/local /usr/local
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /root/.cache /home/mcpuser/.cache

RUN ldconfig && \
    mkdir -p /home/mcpuser/colabfit/models /home/mcpuser/colabfit/datasets /home/mcpuser/colabfit/model_output && \
    chown -R mcpuser:mcpuser /home/mcpuser

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER mcpuser
WORKDIR /home/mcpuser

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import colabfit_mcp; print('ok')" || exit 1

ENTRYPOINT ["entrypoint.sh"]
