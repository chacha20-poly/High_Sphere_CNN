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
from skimage.metrics import structural_similarity as ssim


# 学習済みモデルファイル
model_path = "./checkpoint.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_image(x_clean, x_recon):
    """
    x_clean, x_recon: Tensor (1, C, H, W), [0,1]
    """
    # MSE
    mse_val = F.mse_loss(x_recon, x_clean).item()
    
    # PSNR
    psnr_val = 20 * math.log10(1.0 / math.sqrt(mse_val + 1e-8))
    
    # SSIM
    x_clean_np = x_clean.squeeze(0).cpu().numpy().transpose(1,2,0)
    x_recon_np = x_recon.squeeze(0).cpu().numpy().transpose(1,2,0)
    
    ssim_val = ssim(
        x_clean_np, x_recon_np, 
        data_range=1.0, 
        channel_axis=2,   # チャンネル軸を指定
        win_size=7         # 必要なら奇数に変更
    )
    
    return mse_val, psnr_val, ssim_val
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


model = HD_Autoencoder(in_ch=3, img_size=64).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def load_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # (1, C, H, W)
    return img_tensor.to(device)

# ----------------------
# 推論関数
# ----------------------
def infer(image_path):
    x_clean = load_image(image_path)  # 正解画像
    noise = 0.05 * torch.randn_like(x_clean)
    x_noisy = (x_clean + noise).clamp(0,1)  # ノイズ画像

    with torch.no_grad():
        x_recon = model(x_noisy)

    # Tensor -> CPU -> numpy -> [0,1] clip
    x_noisy_np = x_noisy.squeeze(0).cpu().permute(1,2,0).numpy()
    x_recon_np = x_recon.squeeze(0).cpu().permute(1,2,0).numpy()
    x_recon_np = (x_recon_np - x_recon_np.min()) / (x_recon_np.max() - x_recon_np.min())

    # 可視化
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Input")
    plt.imshow(x_noisy_np)
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.title("Reconstructed")
    plt.imshow(x_recon_np)
    plt.axis("off")
    plt.show()

    # 定量評価
    mse_val, psnr_val, ssim_val = evaluate_image(x_clean, x_recon)
    print(f"MSE: {mse_val:.6f}, PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

for i in range(1,9):
    infer("./data/celeba_images/00001{0}.jpg".format(i))

