import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from scipy.signal import find_peaks
from transformers import BertModel, BertTokenizer

################################################################################
# 1. 모델 아키텍처 정의 (변경 없음)
################################################################################
class HSIDataEncoder(nn.Module):
    def __init__(self, input_dim=341, patch_size=11, embed_dim=128, depth=4, heads=4, out_dim=512):
        super().__init__()
        if input_dim % patch_size != 0:
            raise ValueError(f"input_dim({input_dim}) must be divisible by patch_size({patch_size}).")
        num_patches = input_dim // patch_size
        self.patch_embedding = nn.Linear(patch_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.projector = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.patch_embedding.in_features)
        x = self.patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        cls_output = x[:, 0]
        return self.projector(cls_output)

class TextEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.projector = nn.Linear(768, out_dim)

    def forward(self, text_inputs):
        outputs = self.bert(**text_inputs)
        return self.projector(outputs.pooler_output)

class SpectrumTextCLIP(nn.Module):
    def __init__(self, hsi_input_dim=341):
        super().__init__()
        self.hsi_encoder = HSIDataEncoder(input_dim=hsi_input_dim)
        self.text_encoder = TextEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, hsi_data, text_inputs):
        hsi_features = self.hsi_encoder(hsi_data)
        text_features = self.text_encoder(text_inputs)
        hsi_features = F.normalize(hsi_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return hsi_features, text_features, self.logit_scale.exp()

################################################################################
# 2. 텍스트 생성 함수 및 사용자 정의 데이터셋 (변경 없음)
################################################################################
def generate_descriptive_prompt(spectrum, material_name):
    peaks, properties = find_peaks(spectrum, prominence=np.std(spectrum))
    prompt = f"a spectrum of {material_name}"
    if len(peaks) > 0:
        most_prominent_peak_idx = np.argmax(properties['prominences'])
        peak_location = peaks[most_prominent_peak_idx]
        prompt += f" with a major peak around band {peak_location}"
    return prompt

class NpyClipDataset(Dataset):
    def __init__(self, file_paths, text_labels, tokenizer, target_len=341):
        self.file_paths = file_paths
        self.text_labels = text_labels
        self.tokenizer = tokenizer
        self.target_len = target_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        spectrum_data = np.load(file_path).astype(np.float32)
        if spectrum_data.ndim == 3:
            spectrum_data = np.mean(spectrum_data, axis=(0, 1))
        current_len = spectrum_data.shape[0]
        if current_len < self.target_len:
            padding = np.zeros(self.target_len - current_len, dtype=np.float32)
            spectrum_data = np.concatenate([spectrum_data, padding])
        elif current_len > self.target_len:
            spectrum_data = spectrum_data[:self.target_len]
        text = self.text_labels[idx]
        text_inputs = self.tokenizer(text, padding='max_length', max_length=32, truncation=True, return_tensors="pt")
        text_inputs = {key: val.squeeze(0) for key, val in text_inputs.items()}
        return torch.from_numpy(spectrum_data), text_inputs

################################################################################
# 3. 메인 실행 로직 (수정됨)
################################################################################
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = 'data'
    BANDS_PADDED = 341
    OUTPUT_MODEL_PATH = "clip_model_final.pth"
    LABEL_OUTPUT_PATH = "generated_labels.txt" # <<< 생성된 라벨을 저장할 파일 이름

    CLASS_MAPPING = { "pe": "black polyethylene", "pp": "black polypropylene" }

    print("--- 1. Preparing Dataset with Descriptive Prompts ---")
    
    initial_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
    
    # <<< 수정된 부분: 새로운 리스트에 파일과 라벨을 저장하여 오류 방지
    final_file_list = []
    final_label_list = []
    
    if not initial_files:
        raise FileNotFoundError(f"'{DATA_DIR}' 폴더에 학습할 데이터 파일이 없습니다.")

    print("Generating labels based on spectral features...")
    for full_path in sorted(initial_files):
        prefix = os.path.basename(full_path).split('_')[0]
        if prefix in CLASS_MAPPING:
            material_name = CLASS_MAPPING[prefix]
            spectrum = np.load(full_path).astype(np.float32)
            if spectrum.ndim == 3:
                spectrum = np.mean(spectrum, axis=(0, 1))
            descriptive_label = generate_descriptive_prompt(spectrum, material_name)
            
            final_file_list.append(full_path)
            final_label_list.append(descriptive_label)

    # <<< 새로운 기능: 생성된 라벨을 txt 파일로 저장
    with open(LABEL_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        print("\nGenerated Labels:")
        for i in range(len(final_file_list)):
            file_basename = os.path.basename(final_file_list[i])
            label_text = final_label_list[i]
            print(f"  - File: {file_basename} \n    -> Label: {label_text}")
            f.write(f"File: {file_basename}\nLabel: {label_text}\n---\n")
    print(f"\n✅ Generated labels have been saved to '{LABEL_OUTPUT_PATH}'")

    print("\n--- 2. Setting up model and data loaders ---")
    model = SpectrumTextCLIP(hsi_input_dim=BANDS_PADDED).to(DEVICE)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # <<< 수정된 부분: 최종 리스트를 데이터셋에 사용
    dataset = NpyClipDataset(final_file_list, final_label_list, tokenizer, target_len=BANDS_PADDED)
    data_loader = DataLoader(dataset, batch_size=min(4, len(final_file_list)), shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    print("\n--- 3. Starting Model Training ---")
    model.train()
    for epoch in range(200):
        total_loss = 0
        for hsi_batch, text_batch in data_loader:
            hsi_batch = hsi_batch.to(DEVICE)
            text_batch = {key: val.to(DEVICE) for key, val in text_batch.items()}
            hsi_features, text_features, logit_scale = model(hsi_batch, text_batch)
            logits = logit_scale * hsi_features @ text_features.t()
            labels = torch.arange(len(hsi_batch), device=DEVICE)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch+1:03d}, Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"\n✅ Training complete. Model saved to '{OUTPUT_MODEL_PATH}'")