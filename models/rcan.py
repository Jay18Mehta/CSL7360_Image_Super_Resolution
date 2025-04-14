import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image
import os
from torchvision.transforms.functional import to_tensor, to_pil_image

# === RCAN MODULES ===

class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg_pool(x))
        return x * w


class RCAB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            CALayer(channels)
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualGroup(nn.Module):
    def __init__(self, channels, n_rcab):
        super().__init__()
        modules = [RCAB(channels) for _ in range(n_rcab)]
        modules.append(nn.Conv2d(channels, channels, 3, padding=1))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.body(x)


class Upsampler(nn.Sequential):
    def __init__(self, scale, channels):
        m = []
        for _ in range(int(torch.log2(torch.tensor(scale)))):
            m.append(nn.Conv2d(channels, channels * 4, 3, padding=1))
            m.append(nn.PixelShuffle(2))
        super().__init__(*m)


class RCAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=64, n_rg=10, n_rcab=20, scale=4):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feat, 3, padding=1)
        self.body = nn.Sequential(
            *[ResidualGroup(n_feat, n_rcab) for _ in range(n_rg)],
            nn.Conv2d(n_feat, n_feat, 3, padding=1)
        )
        self.upsample = Upsampler(scale, n_feat)
        self.tail = nn.Conv2d(n_feat, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.upsample(x)
        return self.tail(x)

# === INFERENCE ===

def rcan_upscale(lr_img_pil, model_path="weights/rcan_epoch_20.pth", device='cpu'):
    """
    Super resolves a low-resolution PIL image using the RCAN model.

    Args:
        lr_img_pil (PIL.Image): Low-resolution input image.
        model_path (str): Path to the model weights.
        device (str): 'cuda' or 'cpu'.

    Returns:
        PIL.Image: High-resolution output image.
    """
    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = RCAN(scale=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Convert PIL image to normalized tensor
    lr_tensor = to_tensor(lr_img_pil).unsqueeze(0).to(device)  # Add batch dim

    # Inference
    with torch.no_grad():
        sr_tensor = model(lr_tensor).squeeze(0).clamp(0, 1).cpu()  # Remove batch

    # Convert tensor back to PIL image
    sr_img_pil = to_pil_image(sr_tensor)

    return sr_img_pil


