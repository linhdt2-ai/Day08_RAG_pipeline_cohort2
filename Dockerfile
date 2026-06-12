FROM python:3.10-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết (nếu có)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Sao chép file requirements.txt vào container
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Railway sẽ truyền biến môi trường PORT, mặc định là 8501 nếu chạy local
ENV PORT=8501
EXPOSE $PORT

# Thiết lập lệnh chạy app Streamlit
CMD ["sh", "-c", "streamlit run group_project/app.py --server.port=${PORT} --server.address=0.0.0.0"]
