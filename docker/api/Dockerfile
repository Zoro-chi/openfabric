FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY app/ /app/app/
COPY main.py config.py ./

EXPOSE 8000

CMD ["python", "main.py"]