# step2_vit_mlflow.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from transformers import ViTForImageClassification
import os
import time

# 1. 설정 (GPU 강제 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

if device.type == 'cpu':
    print("⚠️ 경고: GPU가 감지되지 않았습니다! PyTorch가 CUDA 버전으로 설치되었는지 확인하세요.")
else:
    print(f"🔥 GPU 가속 활성화: {torch.cuda.get_device_name(0)}")
    print("   (RTX 5060 Ti의 힘을 보여줘!)")

if not os.path.exists('results'): os.makedirs('results')

# 2. 데이터 준비
# ViT 입력 크기(224x224)에 맞게 변환
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("📥 ViT용 데이터 로드 중...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 3. 모델 정의
def get_vit_model():
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=10,
        ignore_mismatched_sizes=True
    )
    # 🔥 [GPU 전략] Full Fine-tuning
    # 모델의 모든 파라미터를 학습시킴 (정확도 상승 기대)
    return model.to(device)

def train_and_log(seed):
    run_name = f"ViT_Full_Tuning_Seed_{seed}"
    
    with mlflow.start_run(run_name=run_name):
        params = {
            "model_type": "ViT (Full Fine-tuning)",
            "seed": seed,
            "epochs": 1,  # GPU라도 1 Epoch면 충분히 99% 찍음 (MNIST가 쉬워서)
            "batch_size": 64, # 메모리 16GB니까 넉넉하게
            "learning_rate": 2e-5 # Full Tuning일 땐 학습률을 좀 낮게 잡는 게 국룰
        }
        mlflow.log_params(params)
        
        torch.manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        
        model = get_vit_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        print(f"Trial {seed+1}/10 학습 시작...")
        model.train()
        for epoch in range(params['epochs']):
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data).logits
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if i % 10 == 0:
                    print(f"  Step {i}/{len(train_loader)} Loss: {loss.item():.4f}")
            
        # 평가 (전체 데이터 사용)
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).logits
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
        
        # 마지막 시드일 때 모델 저장 (서빙용)
        if seed == 9:
            torch.save(model.state_dict(), "models/vit_model.pth")
            print("💾 서빙용 ViT 모델 저장 완료 (models/vit_model.pth)")

        mlflow.log_metric("test_accuracy", acc)
        print(f"Trial {seed+1}/10 (Seed {seed}) : Accuracy = {acc:.2f}%")
        return acc

if __name__ == "__main__":
    mlflow.set_experiment("MNIST_Hypothesis_Testing")
    vit_accuracies = []
    
    # CUDA 체크
    if not torch.cuda.is_available():
        print("🚨 잠깐! GPU 인식이 안 됩니다. 그냥 돌리면 느려요!")
        print("  -> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("  명령어로 CUDA 버전을 다시 깔아야 할 수도 있습니다.")
    else:
        print("\n🔬 [연구가설 검증] ViT 10회 반복 실험 시작 (GPU Mode On ⚡)...")
        start_time = time.time()
        
        for seed in range(10):
            acc = train_and_log(seed)
            vit_accuracies.append(acc)
        
        end_time = time.time()
        
        np.save("results/vit_accuracies.npy", np.array(vit_accuracies))
        mean_acc = np.mean(vit_accuracies)
        
        print("\n" + "="*40)
        print(f"⏱️ 소요 시간 : {end_time - start_time:.2f}초")
        print(f"✅ ViT 평균 정확도: {mean_acc:.4f}%")
        print("="*40)