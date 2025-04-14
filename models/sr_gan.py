import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from PIL import Image,ImageOps

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_res_blocks=16):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # Upsampling by 2x → 256x256, then another 2x → 512x512
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # 128 → 256
            nn.PReLU(),

            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # 256 → 512
            nn.PReLU()
        )

        self.conv3 = nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.conv1(x)
        res = self.res_blocks(initial)
        res = self.conv2(res)
        out = initial + res
        out = self.upsample(out)
        out = self.conv3(out)
        return torch.clamp(out, 0.0, 1.0)  # to keep output in [0,1]
    

def srgan_upscale(image):
    image = Image.fromarray(image)
    target_size = (128, 128)
    pad_color=(0, 0, 0)
    ImageOps.pad(image, target_size, method=Image.BICUBIC, color=pad_color)
    image = np.array(image)

    image = image/255
    image = torch.from_numpy(image).float().unsqueeze(dim=0).permute(0,3,1,2)

    model = Generator()
    checkpoint = torch.load("weights/model_srgan.pth", map_location=torch.device('cpu'))  # or 'cuda' if using GPU
    state_dict = checkpoint['G_state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove "module." from each key
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    with torch.no_grad():
        output = model(image)

    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 255.0).clip(0, 255).astype("uint8")
    return output 