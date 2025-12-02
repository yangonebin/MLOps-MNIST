# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
from io import BytesIO
import os

# 1. 환경 설정 (CPU 강제 지정)
# 배포 환경의 호환성을 위해 CPU를 사용합니다.
device = torch.device("cpu") 
app = FastAPI()
MODEL = None

# 2. 모델 로딩 함수
def load_model():
    global MODEL 
    
    # 모델 아키텍처 로드
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=10,
        ignore_mismatched_sizes=True
    )
    
    model_path = "models/vit_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
        
    # ⭐ 핵심 변경: map_location=device 
    # (GPU에서 학습한 가중치를 CPU 메모리로 매핑해서 로드함. 필수!)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    MODEL = model
    print("✅ 모델 로딩 완료 (CPU Mode)")


# 3. 서버 시작/종료 이벤트 훅 (모델 로딩 시점 지정)
@app.on_event("startup")
async def startup_event():
    # 서버가 켜지면, 모델 로딩 함수를 실행
    load_model()
    

# 4. 건강 체크 엔드포인트 (Health Check)
@app.get("/health")
def health_check():
    return {"status": "ok", "device": device.type, "model_loaded": MODEL is not None}

# 5. 이미지 전처리를 위한 Transform 정의
# 반드시 step2_vit_mlflow.py 에서 사용한 것과 완전히 동일해야 함!
TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# 6. 예측 엔드포인트 (Prediction Endpoint)
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    사용자가 업로드한 이미지를 ViT 모델로 예측합니다.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")
        
    try:
        # 1. 파일 읽기 및 이미지화
        contents = await file.read() # 업로드된 파일 내용을 비동기적으로 읽음
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # 2. 전처리 및 GPU 배치
        image_tensor = TRANSFORMS(image).unsqueeze(0).to(device)
        
        # 3. 모델 추론 (Inference)
        with torch.no_grad():
            output = MODEL(image_tensor).logits
        
        # 4. 결과 해석
        prediction_index = torch.argmax(output, dim=1).item()
        
        # 5. 결과 반환
        return {
            "filename": file.filename,
            "prediction": int(prediction_index),
            "confidence_score": f"{torch.nn.functional.softmax(output, dim=1)[0][prediction_index].item():.4f}" # 소프트맥스 적용
        }

    except Exception as e:
        # 에러 발생 시 500 에러와 함께 상세 내용 반환
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")