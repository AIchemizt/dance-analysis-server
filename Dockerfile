FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies INCLUDING curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add gevent for async workers
RUN pip install gevent

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Run with Gunicorn using gevent workers
CMD ["gunicorn", "--worker-class", "gevent", "--workers", "2", "--bind", "0.0.0.0:8080", "--timeout", "300", "--access-logfile", "-", "server:app"]