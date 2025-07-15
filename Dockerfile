# استفاده از Python 3.11 slim image
FROM python:3.11-slim

# تنظیم متغیرهای محیطی
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# تنظیم working directory
WORKDIR /app

# نصب system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# کپی کردن requirements و نصب Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# کپی کردن کد پروژه
COPY . .

# ایجاد پوشه‌های مورد نیاز
RUN mkdir -p cache data results logs

# تنظیم مجوزها
RUN chmod +x run_pipeline.py

# اجرای pipeline
CMD ["python", "run_pipeline.py"]