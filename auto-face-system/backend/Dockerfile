# Use InsightFace's official Docker image as base
FROM deepinsight/insightface:latest

WORKDIR /app

# Install FastAPI and additional dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    python-dotenv==1.0.0

# Copy application
COPY main.py .

# Create directory for model cache
RUN mkdir -p /root/.insightface

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-u", "main.py"]
