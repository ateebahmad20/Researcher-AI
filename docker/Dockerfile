# Use an official Python runtime as a parent image
FROM python:3.13.2-slim AS essayAgent

# Specify Python version (optional for documentation)
ENV PYTHON_VERSION=3.13.2

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc python3-dev micro nano libgl1-mesa-glx libglib2.0-0 \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Upgrade pip and install any needed packages specified in requirements.txt
COPY ./docker/requirements.txt ./
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --default-timeout=1000 -r requirements.txt

EXPOSE 5555

COPY ./src .

# Run Uvicorn to serve the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5555"]
