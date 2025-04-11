FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/llm/ /app/llm/

EXPOSE 8000

# Configure huggingface cache
ENV HF_HOME=/root/.cache/huggingface

CMD ["python", "-m", "app.llm.service"]