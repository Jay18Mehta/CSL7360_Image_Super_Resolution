import cv2
import numpy as np

def spline_upscale(image, scale_factor=4):
    """
    Upscale image using Bicubic spline interpolation
    
    Args:
        image: Input image (numpy array or file path)
        scale_factor: Scaling multiplier (default=4)
    
    Returns:
        Upscaled image as numpy array (uint8)
    """
    # Load image if path provided
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image from {image}")
    else:
        img = image.copy()
    
    # Get original dimensions
    h, w = img.shape[:2]
    
    # Calculate new dimensions
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    # Perform bicubic interpolation
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return upscaled