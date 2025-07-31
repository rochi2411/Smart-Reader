# Use the official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_USE_RTLD_GLOBAL=YES

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    poppler-utils \
    build-essential \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_modular.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements_modular.txt

# Copy the application code
COPY src/ ./src/
COPY main.py .
COPY .env .

# Create logs directory
RUN mkdir -p logs

# Expose the port Streamlit will run on
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Command to run the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
