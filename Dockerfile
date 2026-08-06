FROM python:3.11-slim

# Prevent python from writing pyc files and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install dependencies required for librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser

WORKDIR /app

# Install dependencies first
COPY requirements/ ./requirements/
RUN pip install --no-cache-dir -r requirements/prod.txt

# Copy source code with non-root ownership
COPY --chown=appuser:appuser . .

# Create isolated output directory for generated files/images
RUN mkdir -p /app/output /app/media && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]