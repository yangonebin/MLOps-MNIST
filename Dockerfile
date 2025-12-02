# 1. 베이스 이미지: 가벼운 파이썬 (GPU 드라이버 필요 없음)
FROM python:3.11-slim

# 2. 작업 폴더
WORKDIR /app

# 3. 필수 유틸 설치
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. CPU용 PyTorch 설치 (용량 작고 설치 빠름) ⭐
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. 나머지 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 코드 복사
COPY app ./app
COPY models ./models

# 7. 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]