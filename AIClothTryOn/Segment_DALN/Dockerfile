FROM python:3.12-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1


# System deps (optional: add git if loading from repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
&& rm -rf /var/lib/apt/lists/*


WORKDIR /app


# Copy requirements first (better caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir --default-timeout=100 --retries 10 -r requirements.txt


# Copy source
COPY app ./app
COPY model ./model


EXPOSE 80
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]