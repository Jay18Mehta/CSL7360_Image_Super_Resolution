import cv2
import numpy as np

def upsample_lancros(image_x,scale=4):
    h, w = image_x.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)

    # Upsample using Lanczos interpolation
    pred = cv2.resize(image_x, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return pred

def lanczos_kernel(x, a):
    """Lanczos kernel function."""
    if x == 0:
        return 1
    elif -a < x < a:
        return a * np.sinc(x) * np.sinc(x / a)
    else:
        return 0

def upsample_lancros_2(image, scale=4, a=3):
    """
    Upsample an image using Lanczos interpolation.
    
    Parameters:
        image: numpy array (grayscale or RGB)
        scale: scaling factor (default 4x)
        a: size of Lanczos window (default 3)

    Returns:
        Upsampled image as numpy array.
    """
    if image.ndim == 2:  # Grayscale
        h, w = image.shape
        channels = 1
    else:  # RGB
        h, w, channels = image.shape

    new_h, new_w = int(h * scale), int(w * scale)
    output = np.zeros((new_h, new_w, channels)) if channels > 1 else np.zeros((new_h, new_w))

    for y_new in range(new_h):
        for x_new in range(new_w):
            print(y_new,x_new)
            # Map new pixel to original image space
            x_orig = x_new / scale
            y_orig = y_new / scale

            x0 = int(np.floor(x_orig))
            y0 = int(np.floor(y_orig))

            # Accumulators
            pixel = np.zeros(channels) if channels > 1 else 0.0
            norm = 0.0

            for j in range(y0 - a + 1, y0 + a + 1):
                for i in range(x0 - a + 1, x0 + a + 1):
                    if 0 <= i < w and 0 <= j < h:
                        wx = lanczos_kernel(x_orig - i, a)
                        wy = lanczos_kernel(y_orig - j, a)
                        weight = wx * wy
                        if channels > 1:
                            pixel += image[j, i] * weight
                        else:
                            pixel += image[j, i] * weight
                        norm += weight

            if norm > 0:
                output[y_new, x_new] = pixel / norm

    if channels == 1:
        output = output.astype(image.dtype)
    else:
        output = output.clip(0, 255).astype(image.dtype)

    return output