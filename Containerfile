FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy minimal project files (Mounting will handle data/models logic so they persist)
COPY . /app/

# Create necessary directories within the container just in case it runs unmounted
RUN mkdir -p /app/data /app/models/saved /app/output

ENTRYPOINT ["python", "main.py"]
