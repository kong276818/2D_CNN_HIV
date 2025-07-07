import numpy as np
import matplotlib.pyplot as plt

# --- 설정 ---
file_path = 'data/pe_cube_filtered.npy'

# 시각화할 픽셀 및 밴드 인덱스 지정
PIXEL_TO_VIEW = (32, 32)  # (Y, X) 좌표. 이미지 중앙 근처로 자동 설정됩니다.
BAND_TO_VIEW = 100       # 확인할 밴드 번호.

# --- 데이터 로드 및 시각화 ---
try:
    # 1. NumPy 파일 로드
    data_cube = np.load(file_path)
    print(f"데이터 로드 완료. 형태: {data_cube.shape}")

    # 3차원 데이터가 맞는지 확인
    if data_cube.ndim != 3:
        raise ValueError(f"이 코드는 3차원 데이터용입니다. 현재 데이터는 {data_cube.ndim}차원입니다.")

    h, w, c = data_cube.shape
    
    # 2. 시각화할 데이터 준비
    # (1) 평균 스펙트럼
    mean_spectrum = np.mean(data_cube, axis=(0, 1))
    
    # (2) 특정 픽셀 스펙트럼
    y, x = PIXEL_TO_VIEW
    # 지정된 좌표가 이미지 크기를 벗어나지 않도록 조정
    y = min(y, h - 1)
    x = min(x, w - 1)
    pixel_spectrum = data_cube[y, x, :]

    # (3) 특정 밴드 이미지
    band_image = data_cube[:, :, min(BAND_TO_VIEW, c - 1)]

    # 3. 1x3 서브플롯 생성
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle('Hyperspectral Cube Full Visualization', fontsize=16)

    # 플롯 1: 평균 스펙트럼
    axes[0].plot(mean_spectrum)
    axes[0].set_title('1. Mean Spectrum of All Pixels')
    axes[0].set_xlabel('Band Index')
    axes[0].set_ylabel('Average Intensity')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 플롯 2: 특정 픽셀 스펙트럼
    axes[1].plot(pixel_spectrum, color='green')
    axes[1].set_title(f'2. Spectrum at Pixel ({x}, {y})')
    axes[1].set_xlabel('Band Index')
    axes[1].set_ylabel('Intensity')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 플롯 3: 특정 밴드 이미지
    im = axes[2].imshow(band_image, cmap='viridis')
    axes[2].set_title(f'3. Image at Band {BAND_TO_VIEW}')
    axes[2].set_xlabel('Width')
    axes[2].set_ylabel('Height')
    fig.colorbar(im, ax=axes[2], label='Intensity')

    # 4. 레이아웃 최적화 및 그래프 표시
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # suptitle과 겹치지 않도록 조정
    plt.show()


except FileNotFoundError:
    print(f"오류: '{file_path}' 경로에 파일이 없습니다.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")