# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a separate layer
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt


# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install minimal runtime dependencies
# - libglib2.0-0, libsm6, libxext6, libxrender1: Required by OpenCV
# - libfontconfig1: Font rendering for OpenCV
# - ffmpeg: Video codec support
# - curl: Health check tool for container orchestration
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /usr/local /usr/local

# Ensure Python can find installed packages
#ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY analyzer/ ./analyzer/
COPY server.py .

# Create temp directory for uploads
RUN mkdir -p /tmp/dance_uploads

# Non-root user for security (optional but recommended)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /tmp/dance_uploads
USER appuser

# Expose application port
EXPOSE 8080

# Health check for container orchestration
# Checks every 30s, timeout 3s, start checking after 10s, 3 failures = unhealthy
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Production server with Gunicorn
# - gevent workers for async I/O (better for video processing)
# - 2 workers (adjust based on CPU cores: 2-4 * num_cores)
# - 300s timeout for long video processing
# - Access logs to stdout for container log aggregation
CMD ["gunicorn", \
     "--worker-class", "gevent", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "300", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "server:app"]