FROM python:3.11-slim

# Basic setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install vllm
RUN pip install --upgrade pip && \
    pip install vllm[audio] transformers==4.52.4 

# Default command
CMD ["vllm", "serve", "openai/whisper-large-v3", "--host", "0.0.0.0", "--port", "8917", "--trust-remote-code", "--gpu-memory-utilization", "0.5", "--enforce-eager"]
# docker build -t openai-whisper .
# docker run --gpus 1 -p 8883:8917 --name openai-whisper whisper-openai
