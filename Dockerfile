# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy application files
COPY app.py .
# savedmodel.pth must be present in build context to be copied
COPY savedmodel.pth .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "2"]
