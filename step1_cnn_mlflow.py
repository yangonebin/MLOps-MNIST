# step1_cnn_mlflow.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
import os

# 1. 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# 폴더 자동 생성 (여기가 네 걱정을 해결해주는 부분!)
if not os.path.exists('results'): os.makedirs('results')
if not os.path.exists('data'): os.makedirs('data')

# 2. 데이터 준비
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 데이터 다운로드
print("📥 데이터 확인 중...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 3. 모델 정의 (Simple CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_and_log(seed):
    run_name = f"CNN_Trial_Seed_{seed}"
    
    # MLflow 기록 시작 (자동으로 mlruns 폴더가 생김!)
    with mlflow.start_run(run_name=run_name):
        params = {
            "model_type": "CNN",
            "seed": seed,
            "epochs": 3,  # 빠르게 결과 보기 위해 3 epoch
            "batch_size": 128,
            "learning_rate": 0.001
        }
        mlflow.log_params(params)
        
        torch.manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # 학습
        model.train()
        for epoch in range(params['epochs']):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
        # 평가
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
        
        # 결과 기록
        mlflow.log_metric("test_accuracy", acc)
        mlflow.pytorch.log_model(model, "model")
        
        print(f"Trial {seed+1}/10 (Seed {seed}) : Accuracy = {acc:.2f}%")
        return acc

if __name__ == "__main__":
    # 실험 이름 설정
    mlflow.set_experiment("MNIST_Hypothesis_Testing")
    cnn_accuracies = []
    
    print("\n🔬 [MLflow] CNN Baseline 10회 반복 실험 시작...")
    for seed in range(10):
        acc = train_and_log(seed)
        cnn_accuracies.append(acc)
    
    # 결과 저장 (나중에 T-test용)
    np.save("results/cnn_accuracies.npy", np.array(cnn_accuracies))
    
    mean_acc = np.mean(cnn_accuracies)
    print("\n" + "="*40)
    print(f"✅ 평균 정확도: {mean_acc:.4f}%")
    print("="*40)