FROM ai-base:1.0.0

LABEL org.opencontainers.image.title="AI Workstation Python"
LABEL org.opencontainers.image.description="Python tooling layer"
LABEL org.opencontainers.image.version="1.0.0"

# Install uv
ADD https://astral.sh/uv/install.sh /tmp/uv-installer.sh

RUN chmod +x /tmp/uv-installer.sh && \
    /tmp/uv-installer.sh && \
    rm /tmp/uv-installer.sh

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini","--"]

CMD ["bash"]