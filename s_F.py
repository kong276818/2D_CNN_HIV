import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter # 3D Median 필터는 ndimage에 있습니다.
import os

# --- 설정 ---
file_path = 'data/pe.npy'
output_dir = 'data_filtered'
output_file_path_3d = os.path.join(output_dir, 'pe_cube_filtered.npy')

try:
    # --- 1. 3D 데이터 로드 ---
    data_cube = np.load(file_path)
    print(f"원본 3D 큐브 로드 완료. 형태: {data_cube.shape}")

    if data_cube.ndim != 3:
        raise ValueError("입력 데이터가 3차원 HSI 큐브가 아닙니다.")

    # --- 2. 3D 큐브 필터링 ---
    
    # 1단계: 3D Median 필터로 스파이크 제거
    # size=(1, 1, 5) -> 높이(H), 너비(W) 방향은 필터링하지 않고, 밴드(C) 방향으로만 5칸 필터링
    print("1단계: 3D Median 필터링 시작...")
    median_filtered_cube = median_filter(data_cube, size=(1, 1, 5))
    print("Median 필터링 완료.")

    # 2단계: 3D Savitzky-Golay 필터로 최종 스무딩
    # axis=-1 은 마지막 축, 즉 밴드(C) 축을 따라 필터링하라는 의미
    print("2단계: 3D Savitzky-Golay 필터링 시작...")
    final_filtered_cube = savgol_filter(median_filtered_cube, 
                                        window_length=21, 
                                        polyorder=2, 
                                        axis=-1)
    print("Savitzky-Golay 필터링 완료.")

    # --- 3. 필터링된 3D 큐브 저장 ---
    os.makedirs(output_dir, exist_ok=True)
    np.save(output_file_path_3d, final_filtered_cube)
    print(f"✅ 필터링된 3D 큐브를 '{output_file_path_3d}'에 저장했습니다. 형태: {final_filtered_cube.shape}")


    # --- 4. 결과 확인용 시각화 ---
    # 중앙 픽셀의 원본 스펙트럼과 필터링된 스펙트럼 비교
    h, w, c = data_cube.shape
    pixel_y, pixel_x = h // 2, w // 2

    original_spectrum = data_cube[pixel_y, pixel_x, :]
    filtered_spectrum = final_filtered_cube[pixel_y, pixel_x, :]

    plt.figure(figsize=(12, 7))
    plt.plot(original_spectrum, label=f'Original Spectrum at ({pixel_x}, {pixel_y})', color='green', alpha=0.5)
    plt.plot(filtered_spectrum, label=f'Filtered Spectrum at ({pixel_x}, {pixel_y})', color='red', linewidth=2)
    plt.title('Verification: Original vs. 3D Filtered Spectrum')
    plt.xlabel('Band Index')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"오류가 발생했습니다: {e}")