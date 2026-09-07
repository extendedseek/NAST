FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/NAST
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python3 -m pip install --upgrade pip && python3 -m pip install ".[qlora]"

COPY configs ./configs
COPY scripts ./scripts
COPY examples ./examples
ENTRYPOINT ["nast"]
CMD ["--help"]
