# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model weights and inference script
COPY model_weight.pth .
COPY inference.py .

# Expose port for FastAPI
EXPOSE 8000

# Add environment variable for Python buffering
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI application with logging
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]