import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
from tqdm import tqdm
import os

# 1. 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# 2. 캐글 데이터셋 클래스 정의 (CSV -> 이미지 변환)
class KaggleTestDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 픽셀 값 가져오기 (0~255)
        pixel_values = self.data.iloc[idx].values.astype(np.uint8)
        # 2. 1줄짜리 데이터를 28x28 이미지로 변환
        image_array = pixel_values.reshape(28, 28)
        # 3. PIL 이미지로 변환
        image = Image.fromarray(image_array)
        
        if self.transform:
            image = self.transform(image)
            
        return image

# 3. 전처리 정의 (학습 때랑 100% 똑같이!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 4. 데이터 로더 준비
test_path = "data/test.csv"
if not os.path.exists(test_path):
    print(f"🚨 에러: {test_path} 파일이 없습니다! 캐글에서 다운로드해서 data 폴더에 넣으세요.")
    exit()

test_dataset = KaggleTestDataset(csv_file=test_path, transform=transform)
# num_workers=0 (안전빵), batch_size=64 (적당히)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# 5. 모델 로드
print("📥 모델 로딩 중...")
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=10,
    ignore_mismatched_sizes=True
)
# 학습된 가중치 덮어씌우기
model.load_state_dict(torch.load("models/vit_model.pth", map_location=device))
model.to(device)
model.eval()

# 6. 추론 (Inference)
print("🔮 추론 시작...")
predictions = []

with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images).logits
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())

# 7. 제출 파일 생성 (submission.csv)
submission = pd.DataFrame({
    "ImageId": range(1, len(predictions) + 1),
    "Label": predictions
})

submission.to_csv("submission.csv", index=False)
print("\n🎉 완료! 'submission.csv' 파일이 생성되었습니다.")