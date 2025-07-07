import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- 설정 ---
file_path = 'data/pp..npy'

try:
    data_cube = np.load(file_path)
    # 예시를 위해 노이즈가 가장 심한 특정 픽셀의 스펙트럼을 사용합니다.
    h, w, c = data_cube.shape
    original_spectrum = data_cube[h // 2, w // 2, :]

    # --- Savitzky-Golay 필터 적용 ---
    # window_length: 필터링에 사용할 데이터 포인트의 수. 반드시 홀수여야 합니다. 클수록 부드러워집니다.ss
    # polyorder: 피팅에 사용할 다항식의 차수. window_length보다 작아야 합니다.
    window_length = 11  # 창 크기 (홀수)
    polyorder = 2       # 다항식 차수
    
    filtered_spectrum_sg = savgol_filter(original_spectrum, window_length, polyorder)

    # --- 그래프로 원본과 필터링 결과 비교 ---
    plt.figure(figsize=(12, 7))
    
    # 원본 스펙트럼
    plt.plot(original_spectrum, label='Original Noisy Spectrum', color='green', alpha=0.5)
    
    # 필터링된 스펙트럼
    plt.plot(filtered_spectrum_sg, label=f'Savitzky-Golay Filtered (window={window_length}, order={polyorder})', color='red', linewidth=2)
    
    plt.title('Spectrum Filtering using Savitzky-Golay Filter')
    plt.xlabel('Band Index')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

except Exception as e:
    print(f"오류가 발생했습니다: {e}")