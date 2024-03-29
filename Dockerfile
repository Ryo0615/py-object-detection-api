FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN \
    pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]