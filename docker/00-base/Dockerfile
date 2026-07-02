# -----------------------------------------------------------------------------
# AI Workstation Base Image
# -----------------------------------------------------------------------------

FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------

LABEL org.opencontainers.image.title="AI Workstation Base"
LABEL org.opencontainers.image.description="GPU-enabled base image for AI Workstation"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Alok Senapati"
LABEL org.opencontainers.image.vendor="AI Workstation"
LABEL org.opencontainers.image.licenses="MIT"

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# -----------------------------------------------------------------------------
# Install Common Packages
# -----------------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python-is-python3 \
        git \
        curl \
        wget \
        build-essential \
        ca-certificates \
        tini \
        tree \
        htop \
        less \
        unzip \
        zip \
        procps \
        locales \
        vim && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Workspace
# -----------------------------------------------------------------------------

WORKDIR /workspace

# -----------------------------------------------------------------------------
# Default Command
# -----------------------------------------------------------------------------

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["bash"]