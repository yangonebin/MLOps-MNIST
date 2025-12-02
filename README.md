# 🚀 MNIST End-to-End MLOps Pipeline (ViT & Docker)

> **'Data → Modeling → Serving → Deploy' 로 이어지는 MLOps의 전체 사이클을 경험하는 것에 중점을 두었습니다."**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)

---

### 📝 Project Overview (프로젝트 개요)

이 프로젝트는 단순한 MNIST 숫자 분류 모델링을 넘어, **데이터 수집부터 모델 학습, 검증, 그리고 배포까지 이어지는 전체 MLOps 파이프라인**을 구축하는 것을 목표로 합니다.
최신 **Vision Transformer (ViT)** 모델을 도입하여 성능을 고도화하고, **Docker**를 활용하여 어디서든 실행 가능한 배포 환경을 구현했습니다.

👉 **[자세한 개발 과정 및 트러블 슈팅 로그 (Notion)](https://winter-azimuth-dd0.notion.site/Project-Report-MNIST-MLOps-Pipeline-2bd1a506dcad80818476e00e25042394)**

---

### 🐳 How to Run (Docker)

복잡한 설치 없이, 단 한 줄의 명령어로 모델 추론 서버를 실행해볼 수 있습니다.

```bash
docker run -p 8000:8000 yangonebin/mnist-mlops:v1.0 
