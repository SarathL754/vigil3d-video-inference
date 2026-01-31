# 1) Base image (small Python + Linux)
FROM python:3.11-slim

# 2) Prevent Python from writing .pyc files + keep logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3) Set work directory inside container
WORKDIR /app

# 4) Install OS libraries needed for decord/opencv and basic builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 5) Copy only requirements first (better docker cache)
COPY requirements-infer.txt /app/requirements-infer.txt

# 6) Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=120 --retries=5 -r /app/requirements-infer.txt

# 7) Copy your inference app + training model code (needed because infer imports src.models.r3d)
COPY app /app/app
COPY src /app/src

# 8) Expose port for the API
EXPOSE 8000

# 9) Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
