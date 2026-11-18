FROM nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43
ENV DEBIAN_FRONTEND=noninteractive
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv git make g++ libnuma1 libglib2.0-0t64 rsync ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/edmir-python && \
    /opt/edmir-python/bin/pip download --dest /opt/edmir-wheels cmake==3.30.5 numpy==2.2.6 && \
    /opt/edmir-python/bin/pip install --no-index --find-links=/opt/edmir-wheels cmake==3.30.5 numpy==2.2.6
RUN git config --system --add safe.directory /workspace/dependencies/cutlass && \
    git config --system --add safe.directory /workspace/dependencies/nccl
ENV PATH=/opt/edmir-python/bin:/usr/local/cuda/bin:${PATH}
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
