import cv2
import numpy as np

from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image

def fourier_upscale(image, scale_factor=4):
    upscaled_channels = []

    for c in range(3):  # For R, G, B channels
        # Get single channel
        channel = image[:, :, c]

        # Forward FFT
        F = fft2(channel)
        F_shifted = fftshift(F)

        # Zero-padding in frequency domain
        h, w = channel.shape
        new_h, new_w = h * scale_factor, w * scale_factor
        F_padded = np.zeros((new_h, new_w), dtype=complex)

        # Place the original spectrum in the center of the padded one
        h_center, w_center = new_h // 2, new_w // 2
        h_half, w_half = h // 2, w // 2
        F_padded[h_center - h_half:h_center + h_half, w_center - w_half:w_center + w_half] = F_shifted

        # Inverse FFT
        F_unshifted = ifftshift(F_padded)
        upscaled = np.real(ifft2(F_unshifted))

        # ✅ Rescale intensities
        upscaled *= scale_factor ** 2  # Rescale to preserve brightness

        # Normalize to [0, 255]
        upscaled = np.clip(upscaled, 0, 255)
        upscaled_channels.append(upscaled.astype(np.uint8))

    # Stack the 3 upscaled channels
    upscaled_img = np.stack(upscaled_channels, axis=2)
    return upscaled_img