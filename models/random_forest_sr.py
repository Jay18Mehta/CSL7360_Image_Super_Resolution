import numpy as np
from skimage import color, transform, util
from sklearn.ensemble import RandomForestRegressor
from skimage.util import view_as_windows
from PIL import Image

def extract_patches(image, patch_size=(4, 4), step=4):
    patches = view_as_windows(image, patch_size, step=step)
    n_patches = patches.shape[0] * patches.shape[1]
    return patches.reshape(n_patches, -1)

def reconstruct_image_from_patches(patches, image_shape, patch_size=(4, 4), step=4):
    h, w = image_shape
    out = np.zeros((h, w))
    weight = np.zeros((h, w))

    idx = 0
    for i in range(0, h - patch_size[0] + 1, step):
        for j in range(0, w - patch_size[1] + 1, step):
            patch = patches[idx].reshape(patch_size)
            out[i:i+patch_size[0], j:j+patch_size[1]] += patch
            weight[i:i+patch_size[0], j:j+patch_size[1]] += 1
            idx += 1

    return (out / weight)

def random_forest_upscale(lr_image_pil):
    # Convert to grayscale and numpy
    lr_image = np.array(lr_image_pil)
    if lr_image.ndim == 3:
        lr_image = color.rgb2gray(lr_image)

    # Resize to create high-resolution image
    hr_image = transform.rescale(lr_image, 2.0, anti_aliasing=True)
    hr_image = (hr_image * 255).astype(np.uint8)

    # Downscale the HR image again to simulate LR input
    lr_simulated = transform.rescale(hr_image, 0.5, anti_aliasing=True)

    # Prepare data
    patch_size = (4, 4)
    X_train = extract_patches(lr_simulated, patch_size)
    y_train = extract_patches(hr_image, patch_size)

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(X_train, y_train)

    # Predict
    X_test = extract_patches(lr_image, patch_size)
    y_pred = rf.predict(X_test)

    # Reconstruct
    out = reconstruct_image_from_patches(y_pred, hr_image.shape, patch_size)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Return PIL image
    return Image.fromarray(out)
