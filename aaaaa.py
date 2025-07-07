import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    def __init__(self, data_dir):
        # Load HSI
        pe_hsi = np.load(os.path.join(data_dir, 'pe_cube_filtered.npy'))
        pp_hsi = np.load(os.path.join(data_dir, 'pp_cube_filtered.npy'))

        # Load RGB
        pe_rgb = np.load(os.path.join(data_dir, 'pe.npy'))
        pp_rgb = np.load(os.path.join(data_dir, 'pp.npy'))

        # Check shapes and reshape if needed
        pe_hsi = self._ensure_nchw(pe_hsi, is_hsi=True, name="pe_hsi")
        pp_hsi = self._ensure_nchw(pp_hsi, is_hsi=True, name="pp_hsi")
        pe_rgb = self._ensure_nchw(pe_rgb, is_hsi=False, name="pe_rgb")
        pp_rgb = self._ensure_nchw(pp_rgb, is_hsi=False, name="pp_rgb")

        # Optional calibration
        calib_path = os.path.join(data_dir, "calib.txt")
        if os.path.exists(calib_path):
            calib = np.loadtxt(calib_path)
            if calib.ndim == 1 and calib.shape[0] == pe_hsi.shape[1]:
                print("✅ Applying spectral calibration")
                pe_hsi *= calib.reshape(1, -1, 1, 1)
                pp_hsi *= calib.reshape(1, -1, 1, 1)
            else:
                print(f"⚠️ calib.txt shape mismatch: {calib.shape} vs HSI channels {pe_hsi.shape[1]}")
        else:
            print("⚠️ calib.txt not found – skipping spectral correction")

        # --- 🔧 Match spatial dimensions (crop to minimum size) ---
        min_h = min(pe_hsi.shape[2], pp_hsi.shape[2])
        min_w = min(pe_hsi.shape[3], pp_hsi.shape[3])

        pe_hsi = pe_hsi[:, :, :min_h, :min_w]
        pp_hsi = pp_hsi[:, :, :min_h, :min_w]
        pe_rgb = pe_rgb[:, :, :min_h, :min_w]
        pp_rgb = pp_rgb[:, :, :min_h, :min_w]

        # Labels: PE=0, PP=1
        pe_labels = np.zeros(pe_hsi.shape[0], dtype=np.int64)
        pp_labels = np.ones(pp_hsi.shape[0], dtype=np.int64)

        # Concatenate
        self.hsi = np.concatenate([pe_hsi, pp_hsi], axis=0).astype(np.float32)
        self.rgb = np.concatenate([pe_rgb, pp_rgb], axis=0).astype(np.float32)
        self.labels = np.concatenate([pe_labels, pp_labels], axis=0)

        # Normalize
        self.hsi /= np.max(self.hsi)
        self.rgb /= 255.0

    def _ensure_nchw(self, arr, is_hsi=True, name=""):
        """
        Ensure shape is (N, C, H, W)
        HSI: (H, W, B) or (N, H, W, B) → (N, B, H, W)
        RGB: (H, W, 3) or (N, H, W, 3) → (N, 3, H, W)
        """
        if arr.ndim == 3:
            print(f"ℹ️ {name} is single sample 3D – reshaping to batch")
            arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)

        if arr.ndim != 4:
            raise ValueError(f"❌ {name} must be 4D (N, H, W, C), but got shape {arr.shape}")

        arr = np.transpose(arr, (0, 3, 1, 2))  # (N, C, H, W)
        return arr

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.hsi[idx]),
            torch.from_numpy(self.rgb[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )
