# Use a lightweight Python 3.12 image
FROM python:3.12-slim

# Install system dependencies (GCC, python3-dev, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  

# Copy project files
# COPY . .  
COPY ./src /app/src

# Expose FastAPI port
EXPOSE 8000  

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]