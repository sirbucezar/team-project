FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ffmpeg \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Ensure the /app directory is in the Python path
ENV PYTHONPATH=/app

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code
COPY . /app

# Expose port 80 (as expected by Azure Web App)
EXPOSE 80

# Optional: Healthcheck instruction for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl -f http://localhost/health || exit 1

# Use Gunicorn with the default (sync) worker-class
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app.api:app", "--timeout", "120"]