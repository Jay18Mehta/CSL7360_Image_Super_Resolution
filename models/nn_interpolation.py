import cv2
import numpy as np

def nn_upscale(image, scale_factor=4):
    """
    Upscale image using Nearest Neighbor interpolation
    
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
    new_h, new_w = h * scale_factor, w * scale_factor
    
    # Create empty output image
    if len(img.shape) == 3:  # Color image
        upscaled = np.zeros((new_h, new_w, 3), dtype=img.dtype)
    else:  # Grayscale
        upscaled = np.zeros((new_h, new_w), dtype=img.dtype)
    
    # Nearest Neighbor interpolation
    for y in range(new_h):
        for x in range(new_w):
            orig_y = min(int(y / scale_factor), h - 1)
            orig_x = min(int(x / scale_factor), w - 1)
            upscaled[y, x] = img[orig_y, orig_x]
    
    return upscaled