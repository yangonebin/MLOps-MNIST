# latency_test.py
import requests
import time
import io
from PIL import Image
import numpy as np

# 1. 서버 주소
URL = "http://localhost:8000/predict"

# 2. 더미 이미지 생성 (28x28 검은색 이미지)
# 매번 파일 찾기 귀찮으니까 코드에서 바로 만듦
def create_dummy_image():
    img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

# 3. 테스트 설정
NUM_REQUESTS = 50  # 50번 반복
latencies = []

print(f"🚀 Latency Test 시작 ({NUM_REQUESTS}회 요청)...")

try:
    # 워밍업 (첫 번째 요청은 모델 로딩 등으로 느릴 수 있어서 한 번 쏴줌)
    requests.post(URL, files={"file": ("test.png", create_dummy_image(), "image/png")})

    for i in range(NUM_REQUESTS):
        start_time = time.time()
        
        # 요청 전송
        files = {"file": ("test.png", create_dummy_image(), "image/png")}
        response = requests.post(URL, files=files)
        
        end_time = time.time()
        
        if response.status_code == 200:
            # 밀리초(ms) 단위로 변환
            latency = (end_time - start_time) * 1000
            latencies.append(latency)
            print(f"Request {i+1}: {latency:.2f} ms")
        else:
            print(f"Request {i+1}: Failed ({response.status_code})")

    # 4. 결과 계산
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)

    print("\n" + "="*30)
    print(f"📊 테스트 결과 (CPU Inference)")
    print(f"✅ 평균 응답 속도: {avg_latency:.2f} ms")
    print(f"⚡ 최소 응답 속도: {min_latency:.2f} ms")
    print(f"🐢 최대 응답 속도: {max_latency:.2f} ms")
    
    if avg_latency < 100:
        print("🎉 목표 달성! (100ms 미만)")
    else:
        print("⚠️ 목표 미달성 (최적화 필요)")
    print("="*30)

except Exception as e:
    print(f"🚨 에러 발생: 서버가 켜져 있는지 확인하세요! ({e})")