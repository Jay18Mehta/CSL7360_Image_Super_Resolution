import torch
import torch.nn as nn
from collections import OrderedDict
from PIL import Image,ImageOps
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ENCODER — compress from 128 -> 64 -> 32
        self.enc1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)   # 128 → 64
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # 64 → 32

        # DECODER — upsample from 32 → 64 → 128 → 256 → 512
        self.dec1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)   # 32 → 64
        self.dec2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)    # 64 → 128
        self.dec3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)    # 128 → 256
        self.dec4 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)     # 256 → 512

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.relu(self.enc1(x))  # [B, 64, 64, 64]
        x = self.relu(self.enc2(x))  # [B, 128, 32, 32]

        # Decoder
        x = self.relu(self.dec1(x))  # [B, 64, 64, 64]
        x = self.relu(self.dec2(x))  # [B, 32, 128, 128]
        x = self.relu(self.dec3(x))  # [B, 16, 256, 256]
        x = self.sigmoid(self.dec4(x))  # [B, 3, 512, 512]

        return x
    
def autoencoder_upscale(image):
    image = Image.fromarray(image)
    target_size = (128, 128)
    pad_color=(0, 0, 0)
    ImageOps.pad(image, target_size, method=Image.BICUBIC, color=pad_color)
    image = np.array(image)

    image = image/255
    image = torch.from_numpy(image).float().unsqueeze(dim=0).permute(0,3,1,2)

    model = Autoencoder()
    checkpoint = torch.load("weights/model_auto_2.pth", map_location=torch.device('cpu'))  # or 'cuda' if using GPU
    state_dict = checkpoint['state_dict']

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