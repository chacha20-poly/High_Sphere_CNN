import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def show_images(x_noisy, x_recon, x_clean, n=5):
    """ノイズ画像 / 復元画像 / 正解画像 を並べて表示"""
    x_noisy = x_noisy.detach().cpu()
    x_recon = x_recon.detach().cpu()
    x_clean = x_clean.detach().cpu()

    plt.figure(figsize=(9, n*3))
    for i in range(n):
        # 入力 (ノイズ)
        plt.subplot(n, 3, i*3+1)
        plt.imshow(np.transpose(x_noisy[i].numpy(), (1, 2, 0)))
        plt.title("Noisy")
        plt.axis("off")

        # 出力 (復元)
        plt.subplot(n, 3, i*3+2)
        plt.imshow(np.transpose(x_recon[i].numpy(), (1, 2, 0)))
        plt.title("Reconstructed")
        plt.axis("off")

        # 正解 (クリーン画像)
        plt.subplot(n, 3, i*3+3)
        plt.imshow(np.transpose(x_clean[i].numpy(), (1, 2, 0)))
        plt.title("Original")
        plt.axis("off")

    plt.show()


class CelebADataset(Dataset):
    def __init__(self, root, transform=None, noise_std=0.1):
        self.files = sorted(glob.glob(os.path.join(root, "*.jpg")))
        if len(self.files) == 0:
            raise ValueError(f"No images found in {root}")
        self.transform = transform
        self.noise_std = noise_std

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        noisy = img + torch.randn_like(img) * self.noise_std
        noisy = torch.clamp(noisy, 0.0, 1.0)
        return noisy, img  # (入力, 正解)



    

# あなたの HighDimProj (Encoder)
class HighDimCNN(nn.Module):
    def __init__(self, in_ch=3, n_dim=64):
        super().__init__()
        self.in_ch = in_ch
        self.n_dim = n_dim

        # FFT の実部・虚部を結合して 2*in_ch チャンネルにする
        self.conv1 = nn.Conv2d(in_ch*2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d((8,8))

        # 最終特徴ベクトル
        self.fc = nn.Sequential(
            nn.Linear(256*8*8, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

    def forward(self, xr, xi):
        # xr, xi: (B, C, H, W)
        x = torch.cat([xr, xi], dim=1)  # (B, 2*C, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        B, C, H, W = x.shape
        x = x.view(B, -1)
        x = F.normalize(self.fc(x), p=2, dim=1)
        return x  # (B, 256)
# Decoder (MLP)
class CNNDecoder(nn.Module):
    def __init__(self, out_ch=3, feat_dim=256):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 512*4*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_ch, 4, stride=2, padding=1),
            nn.Sigmoid()  # [0,1] 出力
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 512, 4, 4)
        return self.deconv(x)



# 全体モデル (Encoder + Decoder)
class HD_Autoencoder(nn.Module):
    def __init__(self, in_ch=3, img_size=64, n_dim=64):
        super().__init__()
        self.encoder = HighDimCNN(in_ch=in_ch, n_dim=n_dim)
        self.decoder = CNNDecoder(out_ch=in_ch, feat_dim=256)

    def forward(self, x):
        # FFTで実部・虚部を分解
        Xf = torch.fft.fft2(x)
        xr, xi = Xf.real, Xf.imag
        z = self.encoder(xr, xi)
        x_recon = self.decoder(z)
        return x_recon

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HD_Autoencoder(in_ch=3, img_size=64, n_dim=64).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()  # 復元タスクなので MSE が基本

dataset_path = "./data/celeba_images"
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = CelebADataset(dataset_path, transform=transform)
dataset = Subset(dataset, list(range(min(50000, len(dataset)))))
loader = DataLoader(dataset, batch_size=8, shuffle=True)


# dataloader: (x_noisy, x_clean) のペアを返すようにする
for epoch in range(32):
    for x_noisy, x_clean in loader:
        x_noisy, x_clean = x_noisy.to(device), x_clean.to(device)

        # 前向き
        x_recon = model(x_noisy)

        # 損失
        loss = criterion(x_recon, x_clean)

        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss={loss.item():.4f}")

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch
}, "./checkpoint.pth")

print("保存しました")