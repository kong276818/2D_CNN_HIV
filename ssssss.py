import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

# -------------------------------------------------------------------
# Step 0: 가상의 데이터셋 생성 (실제 데이터가 있다면 이 부분은 생략)
# -------------------------------------------------------------------
def create_dummy_dataset(base_dir="dummy_dataset", num_classes=2, samples_per_class=10, depth=16, h=64, w=64):
    print("가상 데이터셋을 생성합니다...")
    for i in range(num_classes):
        class_path = os.path.join(base_dir, f"class_{i}")
        os.makedirs(class_path, exist_ok=True)
        for j in range(samples_per_class):
            sample_path = os.path.join(class_path, f"sample_{j}")
            os.makedirs(sample_path, exist_ok=True)
            for d in range(depth):
                # 흑백 이미지 생성
                dummy_image = Image.new('L', (w, h), color=int(255 * (i+1) / num_classes))
                dummy_image.save(os.path.join(sample_path, f"slice_{d:02d}.png"))
    print("데이터셋 생성 완료.")

# -------------------------------------------------------------------
# Step 1: 데이터 로딩을 위한 Dataset 클래스 정의
# -------------------------------------------------------------------
class Custom3DDataset(Dataset):
    def __init__(self, base_dir, depth, height, width):
        self.base_dir = base_dir
        self.depth = depth
        self.height = height
        self.width = width
        
        self.samples = []
        self.labels = []
        
        # 클래스 폴더 순회 (class_0, class_1, ...)
        for label, class_name in enumerate(sorted(os.listdir(base_dir))):
            class_path = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            # 샘플 폴더 순회 (sample_0, sample_1, ...)
            for sample_name in sorted(os.listdir(class_path)):
                sample_path = os.path.join(class_path, sample_name)
                if os.path.isdir(sample_path):
                    self.samples.append(sample_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        label = self.labels[idx]
        
        # 이전 코드에서 작성한 이미지 스택 처리 로직 활용
        file_list = sorted([os.path.join(sample_path, f) for f in os.listdir(sample_path)])
        files_to_process = file_list[:self.depth]
        
        volume = []
        for file_path in files_to_process:
            with Image.open(file_path) as img:
                img_resized = img.resize((self.width, self.height))
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                volume.append(img_array)
        
        # (D, H, W) 형태로 쌓기
        volume_stack = np.stack(volume, axis=0)
        # 채널(C) 차원 추가 (흑백이므로 1)
        volume_with_channel = np.expand_dims(volume_stack, axis=0) # Shape: (1, D, H, W)
        
        return torch.from_numpy(volume_with_channel), torch.tensor(label, dtype=torch.long)

# -------------------------------------------------------------------
# Step 2: 3D CNN 모델 정의
# -------------------------------------------------------------------
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        # MaxPool3d를 두 번 적용했으므로 D, H, W가 각각 4분의 1로 줄어듦
        # D=16->4, H=64->16, W=64->16
        self.fc_layer = nn.Linear(32 * (16 // 4) * (64 // 4) * (64 // 4), num_classes)

    def forward(self, x):
        # Input shape: (N, 1, D, H, W)
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.flatten(out)
        out = self.fc_layer(out)
        return out

# -------------------------------------------------------------------
# Step 3: 학습 파라미터 설정 및 실행
# -------------------------------------------------------------------
if __name__ == '__main__':
    # --- 하이퍼파라미터 ---
    DATA_DIR = "dummy_dataset"
    NUM_CLASSES = 2
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4  # 일반적으로 안정적인 학습률
    EPOCHS = 10
    
    # --- 데이터 차원 ---
    DEPTH, HEIGHT, WIDTH = 16, 64, 64

    # 1. 가상 데이터셋 생성
    create_dummy_dataset(base_dir=DATA_DIR, num_classes=NUM_CLASSES, depth=DEPTH, h=HEIGHT, w=WIDTH)

    # 2. 데이터셋 및 데이터로더 준비
    train_dataset = Custom3DDataset(base_dir=DATA_DIR, depth=DEPTH, height=HEIGHT, width=WIDTH)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 모델, Loss 함수, Optimizer 정의
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = Simple3DCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()  # 분류 문제에 적합한 Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 학습 루프
    print("학습을 시작합니다...")
    model.train() # 모델을 학습 모드로 설정

    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # --- 기본 학습 5단계 ---
            # 1. 순전파 (Forward pass)
            outputs = model(inputs)
            
            # 2. Loss 계산
            loss = criterion(outputs, labels)
            
            # 3. Gradient 초기화
            optimizer.zero_grad()
            
            # 4. 역전파 (Backward pass)
            loss.backward()
            
            # 5. 가중치 업데이트
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    print("학습 완료.")