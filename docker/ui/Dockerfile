FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the UI application code
COPY ui/ /app/ui/

EXPOSE 7860

CMD ["python", "-m", "ui.app"]