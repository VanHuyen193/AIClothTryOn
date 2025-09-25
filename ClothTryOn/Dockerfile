FROM python:3.9-slim

# Ngăn python sinh ra file .pyc và log buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cài đặt dependencies của hệ thống
RUN apt-get update \
    && apt-get install -y gcc default-libmysqlclient-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục project
WORKDIR /code

# Copy requirements
COPY requirements.txt /code/

# Cài Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy toàn bộ source code
COPY . /code/
