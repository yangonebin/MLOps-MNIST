# step3_hypothesis_test.py
import numpy as np
from scipy import stats
import os

print("📊 [Step 3] 가설 검증: 통계적 가정 확인 및 검정 수행...")

# 1. 데이터 로드
cnn_path = "results/cnn_accuracies.npy"
vit_path = "results/vit_accuracies.npy"

if not os.path.exists(cnn_path) or not os.path.exists(vit_path):
    print("🚨 데이터 파일이 없습니다! Step 1, 2를 먼저 실행하세요.")
    exit()

cnn_acc = np.load(cnn_path)
vit_acc = np.load(vit_path)

print(f"\n📈 [Descriptive Stats]")
print(f" - CNN Mean: {np.mean(cnn_acc):.4f}% (std: {np.std(cnn_acc):.4f})")
print(f" - ViT Mean: {np.mean(vit_acc):.4f}% (std: {np.std(vit_acc):.4f})")

# 2. 정규성 검정 (Shapiro-Wilk Test)
# 귀무가설: 데이터가 정규분포를 따른다. (P > 0.05면 정규성 만족)
print(f"\n1️⃣ 정규성 검정 (Shapiro-Wilk)")
shapiro_cnn = stats.shapiro(cnn_acc)
shapiro_vit = stats.shapiro(vit_acc)

print(f" - CNN Normality P-value: {shapiro_cnn.pvalue:.4f}")
print(f" - ViT Normality P-value: {shapiro_vit.pvalue:.4f}")

is_normal = (shapiro_cnn.pvalue > 0.05) and (shapiro_vit.pvalue > 0.05)

if is_normal:
    print("👉 두 집단 모두 정규성을 만족합니다. (Parametric Test 진행)")
    
    # 3. 등분산성 검정 (Levene's Test) - 정규성 만족 시 수행
    # 귀무가설: 두 집단의 분산이 같다.
    print(f"\n2️⃣ 등분산성 검정 (Levene's Test)")
    levene = stats.levene(cnn_acc, vit_acc)
    print(f" - Levene P-value: {levene.pvalue:.4f}")
    
    if levene.pvalue > 0.05:
        print("👉 등분산성을 만족합니다. (Student's T-test)")
        t_stat, p_value = stats.ttest_ind(vit_acc, cnn_acc, equal_var=True, alternative='greater')
        test_name = "Student's T-test"
    else:
        print("👉 등분산성을 만족하지 않습니다. (Welch's T-test)")
        t_stat, p_value = stats.ttest_ind(vit_acc, cnn_acc, equal_var=False, alternative='greater')
        test_name = "Welch's T-test"

else:
    print("👉 정규성을 만족하지 못하는 집단이 있습니다. (Non-parametric Test 진행)")
    
    # 3-Alt. 비모수 검정 (Mann-Whitney U Test)
    print(f"\n2️⃣ 비모수 검정 (Mann-Whitney U Test)")
    # alternative='greater': ViT가 CNN보다 큰지 검정
    u_stat, p_value = stats.mannwhitneyu(vit_acc, cnn_acc, alternative='greater')
    test_name = "Mann-Whitney U Test"

# 4. 최종 결론
print("\n" + "="*50)
print(f"🧪 최종 검정 결과 ({test_name})")
print(f" - P-value : {p_value:.4e}")
print("="*50)

alpha = 0.05
if p_value < alpha:
    print(f"✅ P-value < {alpha}")
    print("🎉 결론: 귀무가설 기각! ViT 모델이 통계적으로 유의미하게 더 우수합니다.")
else:
    print(f"❌ P-value >= {alpha}")
    print("결론: 귀무가설 기각 실패. 통계적으로 유의미한 차이가 없습니다.")