# cardstream web client.
#
# SECURITY: this image binds 0.0.0.0 (required inside a container) and the app
# is UNAUTHENTICATED while holding a paid Ximilar key. Publish the port to
# loopback only, never to a LAN or the internet:
#
#   docker build -t cardstream .
#   docker run --rm -e XIMILAR_API_KEY -p 127.0.0.1:8001:8001 \
#     -v cardstream-models:/models cardstream
#
# Model weights are fetched into /models on first run (CARDSTREAM_MODELS_URL);
# mount a pre-populated directory (-v ./models:/models) to skip the fetch.
FROM python:3.12-slim

# libglib2.0-0/libgomp1: opencv-python-headless runtime; ffmpeg: --listen /
# --ffmpeg sources (OBS pushing RTMP/SRT — the main containerized use case).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 curl ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[client,onnx]'

COPY docker/entrypoint.sh /usr/local/bin/cardstream-entrypoint
RUN chmod +x /usr/local/bin/cardstream-entrypoint \
    && useradd --create-home cardstream \
    && mkdir /models && chown cardstream /models
USER cardstream

ENV CARDSTREAM_MODELS_URL="https://cardstream.ai/models/cardstream-models-v1.tar.gz"
VOLUME /models
EXPOSE 8001
ENTRYPOINT ["cardstream-entrypoint"]
