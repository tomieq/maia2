# Use a slim Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for some Python packages (and for torch wheels)
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps first to leverage Docker cache
COPY maia2/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt
RUN pip install maia2
# Copy the application code and model files into the image.
# This includes maia2/ and maia2_models/ (config + rapid_model.pt)
COPY maia2/*.py /app

# Expose the REST port
EXPOSE 8080

# Default command: run uvicorn serving the FastAPI app defined in maia2/rest.py
CMD ["uvicorn", "maia2.rest:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]