import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
from torchvision.io import read_image, ImageReadMode

# ===============================
# ✅ 1. 설정 클래스
# ===============================
class Config:
    """하이퍼파라미터 및 경로 설정을 위한 클래스"""
    DATA_DIR = "dummy_dataset"  # 데이터를 읽어올 폴더 이름
    BATCH_SIZE = 4
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    SAVE_PATH = "best_model_image_from_folders.pth"
    TRAIN_SPLIT = 0.7
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# ✅ 2. 데이터셋 클래스 (이미지 폴더 기반)
# ===============================
class ImageFolderDataset(Dataset):
    """지정된 폴더 구조에서 이미지(.png, .jpg 등) 데이터를 로드하고 전처리하는 클래스"""
    def __init__(self, data_dir):
        self.image_files = []
        self.labels = []
        
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"❌ 데이터 디렉토리 '{data_dir}'를 찾을 수 없습니다.")

        self.class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if not self.class_names:
            raise ValueError(f"'{data_dir}' 안에 클래스 폴더(예: class_0)가 없습니다.")

        print(f"🔍 클래스 발견: {self.class_names}")
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}

        for label_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_dir, class_name)
            # 하위 폴더까지 모두 탐색
            for root, _, filenames in os.walk(class_path):
                for filename in filenames:
                    if os.path.splitext(filename)[1].lower() in image_extensions:
                        self.image_files.append(os.path.join(root, filename))
                        self.labels.append(label_idx)

        if not self.image_files:
            raise ValueError(f"'{data_dir}'의 하위 폴더에서 이미지 파일을 찾을 수 없습니다.")

        # 모든 이미지의 H, W 크기를 맞추기 위해 첫 번째 이미지 기준으로 로드
        print("데이터를 로드하여 크기를 맞추는 중입니다...")
        images_as_tensors = [self._load_and_process_image(path) for path in tqdm(self.image_files, desc="Loading images")]
        
        min_h = min(img.shape[1] for img in images_as_tensors)
        min_w = min(img.shape[2] for img in images_as_tensors)
        
        # 데이터를 최종 Tensor로 변환
        processed_images = [img[:, :min_h, :min_w] for img in images_as_tensors]
        self.images = torch.stack(processed_images)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        print(f"✅ 이미지 데이터 로드 완료: Images {self.images.shape}, Labels {self.labels.shape}")

    def _load_and_process_image(self, path):
        # 이미지를 RGB로 읽고 0-1 사이의 float 값으로 변환
        img = read_image(path, mode=ImageReadMode.RGB)
        return img.float() / 255.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ===============================
# ✅ 3. 모델 정의 (이미지 분류기)
# ===============================
class ImageClassifier(nn.Module):
    """일반 이미지 특징으로 분류하는 CNN 모델"""
    def __init__(self, num_classes=2):
        super().__init__()
        # 3채널(RGB) 이미지를 입력으로 받음
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, img):
        features = self.feature_extractor(img)
        output = self.classifier(features)
        return output

# ===============================
# ✅ 4. 학습, 검증, 테스트 함수
# ===============================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    if not dataloader: return 0.0, 0.0
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not dataloader: return 0.0, 0.0
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds)

def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if not all_labels:
        print("⚠️ 경고: 테스트 데이터가 없어 성능 측정을 건너뜁니다.")
        return

    print("\n--- 📊 최종 성능 평가 ---")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# ===============================
# ✅ 5. 메인 실행 함수
# ===============================
def main():
    cfg = Config()
    print(f"🚀 Using device: {cfg.DEVICE}")

    full_dataset = ImageFolderDataset(cfg.DATA_DIR)
    dataset_size = len(full_dataset)

    test_size = int(cfg.TEST_SPLIT * dataset_size)
    val_size = int(cfg.VALIDATION_SPLIT * dataset_size)
    train_size = dataset_size - val_size - test_size

    if val_size <= 0 or test_size <= 0 or train_size <= 0:
        raise ValueError(f"데이터셋 크기({dataset_size}개)가 너무 작아 학습/검증/테스트 분할을 할 수 없습니다. 더 많은 데이터가 필요합니다.")

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    print(f"Split: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

    model = ImageClassifier(num_classes=len(full_dataset.class_names)).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

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

    if os.path.exists(cfg.SAVE_PATH):
        print(f"\n🔬 최적 모델({cfg.SAVE_PATH})을 불러와 최종 성능을 측정합니다...")
        best_model = ImageClassifier(num_classes=len(full_dataset.class_names)).to(cfg.DEVICE)
        best_model.load_state_dict(torch.load(cfg.SAVE_PATH))
        evaluate_model(best_model, test_loader, cfg.DEVICE, full_dataset.class_names)
    else:
        print("\n⚠️ 저장된 최적 모델이 없습니다. 테스트를 건너뜁니다.")


def prepare_dummy_data(data_dir="dummy_dataset"):
    """실행에 필요한 더미 폴더와 이미지(.png) 데이터를 생성합니다."""
    if os.path.exists(data_dir):
        return
    
    print(f"ℹ️ '{data_dir}' 폴더와 더미 이미지 데이터를 생성합니다.")
    os.makedirs(data_dir, exist_ok=True)
    
    for i in range(2): # class_0, class_1
        class_path = os.path.join(data_dir, f"class_{i}")
        os.makedirs(class_path, exist_ok=True)
        for j in range(20): # 20개 샘플
            # 3채널(RGB)의 64x64 크기 더미 이미지 생성
            dummy_image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
            torchvision.utils.save_image(dummy_image.float()/255.0, os.path.join(class_path, f"sample_{j}.png"))

if __name__ == "__main__":
    # prepare_dummy_data() # 로컬에서 테스트 시 주석 해제하여 더미 데이터 생성
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        print(f"\n--- 🚨 오류 발생! ---")
        print(e)
        print("\n--- 💡 해결 방법 ---")
        print(f"1. '{Config.DATA_DIR}' 폴더가 있는지 확인하세요.")
        print(f"2. '{Config.DATA_DIR}' 폴더 안에 'class_0', 'class_1'과 같은 하위 폴더가 있는지 확인하세요.")
        print("3. 각 클래스 폴더 또는 그 하위 폴더에 .png, .jpg 등의 이미지 파일이 있는지 확인하세요.")