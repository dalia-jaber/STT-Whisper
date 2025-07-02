FROM python:3.11-slim

# Basic setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --upgrade pip && \
    pip install \
        vllm[audio] \
        transformers==4.52.4 \
        fastapi \
        uvicorn[standard] \
        pyyaml

# Default command
CMD ["python", "app.py"]
# docker build -t openai-whisper .
# docker run --gpus 1 -p 8883:8917 --name openai-whisper whisper-openai
