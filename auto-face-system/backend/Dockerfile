FROM python:3.10-slim

WORKDIR /app

# Install ALL system dependencies INCLUDING BUILD TOOLS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    build-essential \
    g++ \
    gcc \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install packages in SPECIFIC ORDER to avoid conflicts
RUN pip install --no-cache-dir numpy==1.24.3

# Install onnxruntime SEPARATELY (most critical)
RUN pip install --no-cache-dir onnxruntime==1.16.3

# Install other dependencies
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78
RUN pip install --no-cache-dir scipy==1.11.4
RUN pip install --no-cache-dir Pillow==10.1.0

# Install InsightFace (depends on above packages AND build tools)
RUN pip install --no-cache-dir insightface==0.7.3

# Install FastAPI with COMPATIBLE pydantic version
RUN pip install --no-cache-dir \
    pydantic>=2.9.2 \
    pydantic-settings>=2.6.0 \
    fastapi>=0.115.0 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6 \
    python-dotenv==1.0.0

# Copy application
COPY main.py .

# Create directory for model cache
RUN mkdir -p /root/.insightface

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-u", "main.py"]
