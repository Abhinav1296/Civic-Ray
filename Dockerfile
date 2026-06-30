FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY setup_nltk.py .
RUN python setup_nltk.py

COPY . .

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app.py"]