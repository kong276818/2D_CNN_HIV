# 🧪 2D_CNN_HIV: HSI + RGB 기반 미세 플라스틱 분류 모델

본 프로젝트는 HSI(Hyperspectral Imaging)와 RGB 이미지를 활용하여 검정색 미세 플라스틱 (PE / PP)을 분류하는 **2D CNN 기반 경량 분류 모델**입니다.  
MWIR 대역의 초분광 데이터를 기반으로 하며, 실시간 적용을 고려한 경량 구조로 설계되었습니다.

---

## 🔍 주요 특징

- ✅ HSI + RGB 멀티모달 입력 지원
- ✅ PE (폴리에틸렌) / PP (폴리프로필렌) 이진 분류
- ✅ 2D CNN 기반 듀얼 브랜치 구조
- ✅ 실시간 추론 가능
- ✅ 전처리 포함 (자동 차원 정렬 및 정규화)

---

## 🗂️ 디렉터리 구조
## 📊 모델 성능

> ✅ **혼동 행렬 예시**

| 실제 \ 예측 | class_0 (PE) | class_1 (PP) |
|-------------|--------------|--------------|
| class_0     | 28           | 0            |
| class_1     | 0            | 20           |

- 정확도: **100%** (예시 기준)
- 평가 방식: train-validation split & accuracy

---

## 🧠 모델 구조 요약

- `hsi_branch`: (N, HSI_C, H, W) 입력 → Conv2D → AvgPool
- `rgb_branch`: (N, 3, H, W) 입력 → Conv2D → AvgPool
- `concat` → Linear → 분류 출력

```python
class HSI_RGB_Classifier(nn.Module):
    def forward(self, hsi, rgb):
        h_feat = self.hsi_branch(hsi)
        r_feat = self.rgb_branch(rgb)
        combined = torch.cat([h_feat, r_feat], dim=1)
        return self.classifier(combined)

🚀 실행 방법
1. 데이터 디렉터리 구성
bash
복사
편집
/data
├── pe_cube_filtered.npy
├── pp_cube_filtered.npy
├── pe.npy
├── pp.npy
2. 학습 실행
bash
복사
편집
python HSI_RGB_BI_cLIP.py
3. 학습 완료 후 모델 저장
model_hsi_rgb.pth가 저장됩니다.

💬 문의
문의사항이나 제안사항은 Issues 또는 Pull Request로 자유롭게 남겨주세요.

![aaaaa](https://github.com/user-attachments/assets/938ff3a1-3f78-46e1-abcc-203927494ea8)

