import torch
import torch.nn as nn
from collections import OrderedDict

# SRCNN Model Definition 
class SuperResolutionCNN(nn.Module):
    def __init__(self):
        super(SuperResolutionCNN, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU()
        )
        
        # Upsampling blocks
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        
        # Reconstruction
        self.reconstruction = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        
    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        x = self.reconstruction(x)
        return torch.sigmoid(x)
    

def srcnn_upscale(image, model_path="model_cnn.pth", device='cpu'):
    """
    Upscale image using SRCNN model
    Input: numpy array (H,W,3) in range 0-255
    Output: numpy array (H,W,3) in range 0-255
    
    Args:
        image: Input image as numpy array (H,W,3) in range 0-255
        model_path: Path to the trained SRCNN model weights
        device: 'cpu' or 'cuda' for GPU acceleration
    """
    # Normalize and convert to tensor [1,3,H,W]
    image = image / 255.0
    image = torch.from_numpy(image).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
    
    # Load model
    model = SuperResolutionCNN().to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint  # Assume direct state dict
    
    # Remove "module." prefix if present (for DataParallel models)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Process image
    with torch.no_grad():
        output = model(image)
    
    # Convert back to numpy array [H,W,3] 0-255
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 255.0).clip(0, 255).astype("uint8")
    
    return output
