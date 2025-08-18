FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install essential OS-level dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY . .
RUN pip install -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=7373
ENV HOSTNAME=0.0.0.0

# Set CUDA-related environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Default command - will be overridden by specific environment Dockerfiles
CMD ["python", "main_docker.py"]
