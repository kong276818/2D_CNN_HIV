import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
import time
from tqdm import tqdm

# ===============================
# ✅ 1. 설정 클래스
# ===============================
class Config:
    """하이퍼파라미터 및 경로 설정을 위한 클래스"""
    DATA_DIR = "data"
    BATCH_SIZE = 4
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    SAVE_PATH = "best_model_hsi_only.pth"  # HSI 전용 모델 저장 경로
    VALIDATION_SPLIT = 0.2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# ✅ 2. 데이터셋 클래스 (HSI 전용)
# ===============================
class HSIDataset(Dataset):
    """HSI(초분광) 데이터를 로드하고 전처리하는 클래스"""
    def __init__(self, data_dir):
        try:
            # HSI 데이터만 로드합니다.
            pe_hsi = np.load(os.path.join(data_dir, 'pe_cube_filtered.npy'))
            pp_hsi = np.load(os.path.join(data_dir, 'pp_cube_filtered.npy'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ '{e.filename}' 파일을 찾을 수 없습니다. '{data_dir}' 폴더에 HSI 파일들이 있는지 확인하세요.")

        # 데이터를 PyTorch 형식 (N, C, H, W)으로 변환합니다.
        pe_hsi = self._to_nchw(pe_hsi, name='pe_hsi')
        pp_hsi = self._to_nchw(pp_hsi, name='pp_hsi')

        # 이미지 크기를 맞춥니다.
        min_h = min(pe_hsi.shape[2], pp_hsi.shape[2])
        min_w = min(pe_hsi.shape[3], pp_hsi.shape[3])
        pe_hsi = pe_hsi[:, :, :min_h, :min_w]
        pp_hsi = pp_hsi[:, :, :min_h, :min_w]

        # 데이터와 레이블을 결합하고 정규화합니다.
        self.hsi = np.concatenate([pe_hsi, pp_hsi], axis=0).astype(np.float32)
        pe_labels = np.zeros(pe_hsi.shape[0], dtype=np.int64)
        pp_labels = np.ones(pp_hsi.shape[0], dtype=np.int64)
        self.labels = np.concatenate([pe_labels, pp_labels], axis=0)

        # 정규화
        self.hsi /= (np.max(self.hsi) if np.max(self.hsi) > 0 else 1.0)

        self.hsi_channels = self.hsi.shape[1]
        print(f"✅ HSI 데이터 로드 완료: HSI {self.hsi.shape}, Labels {self.labels.shape}")

    def _to_nchw(self, arr, name=''):
        """Numpy 배열을 PyTorch가 선호하는 (N, C, H, W) 형태로 변환합니다."""
        if arr.ndim == 3: arr = arr[np.newaxis, ...]
        if arr.ndim != 4: raise ValueError(f"❌ {name}은 4차원 배열이어야 합니다. 현재 형태: {arr.shape}")
        
        # 채널이 마지막 차원에 있을 경우 (N, H, W, C) -> (N, C, H, W)
        if arr.shape[-1] < arr.shape[1]:
             arr = np.transpose(arr, (0, 3, 1, 2))
        return arr

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # HSI 데이터와 레이블만 반환합니다.
        return (torch.from_numpy(self.hsi[idx]), torch.tensor(self.labels[idx], dtype=torch.long))

# ===============================
# ✅ 3. 모델 정의 (HSI 전용)
# ===============================
class HSI_Classifier(nn.Module):
    """HSI 특징만으로 분류하는 CNN 모델"""
    def __init__(self, hsi_channels, num_classes=2):
        super().__init__()
        # HSI 특징 추출을 위한 CNN 브랜치
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(hsi_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        # 최종 분류를 위한 분류기
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, hsi):
        features = self.feature_extractor(hsi)
        output = self.classifier(features)
        return output

# ===============================
# ✅ 4. 학습 및 검증 함수
# ===============================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """한 에포크 동안 모델을 학습합니다."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for hsi, labels in tqdm(dataloader, desc="Training"):
        hsi, labels = hsi.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(hsi) # 모델에 HSI 데이터만 전달
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # (수정) 데이터로더가 비어있는 경우를 처리합니다.
    if len(dataloader) == 0:
        return 0.0, 0.0
        
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def validate_one_epoch(model, dataloader, criterion, device):
    """한 에포크 동안 모델 성능을 검증합니다."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for hsi, labels in tqdm(dataloader, desc="Validation"):
            hsi, labels = hsi.to(device), labels.to(device)
            outputs = model(hsi) # 모델에 HSI 데이터만 전달
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # (수정) 데이터로더가 비어있는 경우를 처리합니다.
    if len(dataloader) == 0:
        print("⚠️ 경고: 검증 데이터로더가 비어있어 검증을 건너뜁니다.")
        return 0.0, 0.0

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# ===============================
# ✅ 5. 메인 실행 함수
# ===============================
def main():
    """전체 학습 과정을 조율하고 실행합니다."""
    cfg = Config()
    print(f"🚀 Using device: {cfg.DEVICE}")

    # --- 데이터 준비 ---
    full_dataset = HSIDataset(cfg.DATA_DIR)
    
    # 데이터셋이 비어있지 않을 때만 분할을 시도합니다.
    if len(full_dataset) > 0:
        val_size = int(len(full_dataset) * cfg.VALIDATION_SPLIT)
        train_size = len(full_dataset) - val_size
        
        # val_size가 0이 되지 않도록 최소 1개는 할당 (단, 전체 샘플이 1개 이상일 때)
        if train_size > 0 and val_size == 0:
            val_size = 1
            train_size = len(full_dataset) - val_size

        if train_size <= 0 or val_size <= 0:
            train_dataset, val_dataset = full_dataset, None # 한 쪽이 0이면 분할하지 않음
            print(f"⚠️ 경고: 데이터셋 크기가 너무 작아({len(full_dataset)}개) 학습/검증 분할을 건너뜁니다.")
        else:
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        train_dataset, val_dataset = full_dataset, None

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    # val_dataset이 None이 아닐 경우에만 DataLoader 생성
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False) if val_dataset else []
    
    print(f"Split: {len(train_dataset)} train, {len(val_dataset) if val_dataset else 0} validation samples.")

    # --- 모델, 손실함수, 옵티마이저 초기화 ---
    model = HSI_Classifier(hsi_channels=full_dataset.hsi_channels).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # --- 학습 루프 ---
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(cfg.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{cfg.EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, cfg.DEVICE)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.SAVE_PATH)
            print(f"🎉 New best model saved with accuracy: {best_val_acc:.4f} to {cfg.SAVE_PATH}")

    total_time = time.time() - start_time
    print(f"\n✨ Training complete in {total_time:.2f}s. Best validation accuracy: {best_val_acc:.4f}")

def prepare_dummy_data():
    """실행에 필요한 더미 HSI 데이터와 폴더를 생성합니다."""
    if not os.path.exists("data"):
        print("ℹ️ 'data' 폴더와 더미 HSI 데이터를 생성합니다. 실제 데이터로 교체해야 합니다.")
        os.makedirs("data")
        dummy_pe_hsi = np.random.rand(10, 100, 100, 361).astype(np.float32)
        dummy_pp_hsi = np.random.rand(10, 100, 100, 361).astype(np.float32)
        np.save("data/pe_cube_filtered.npy", dummy_pe_hsi)
        np.save("data/pp_cube_filtered.npy", dummy_pp_hsi)

if __name__ == "__main__":
    prepare_dummy_data()
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        print(f"\n--- 🚨 오류 발생! ---")
        print(e)
        print("\n--- 💡 해결 방법 ---")
        print("1. 'data' 폴더 안에 2개의 .npy 파일(pe_cube_filtered, pp_cube_filtered)이 모두 있는지 확인하세요.")
        print("2. 오류 메시지에 언급된 파일이 올바른 형태(shape)를 가지고 있는지 확인하세요.")