import numpy as np
from skimage import transform, util, color
from sklearn.ensemble import RandomForestRegressor
from skimage.util import view_as_windows
from PIL import Image

# CONFIGURATION
PATCH_SIZE = (3, 3)
STEP = 1
N_ESTIMATORS = 10
MAX_DEPTH = 10
SCALE_FACTOR = 2
SAMPLE_PATCHES = 10000  # Controls speed/accuracy trade-off

def extract_patches(img, patch_size, step):
    patches = view_as_windows(img, patch_size, step)
    h, w = patches.shape[:2]
    return patches.reshape(h * w, -1)

def train_rf(X, y):
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        n_jobs=-1
    )
    rf.fit(X, y)
    return rf

def predict_and_reconstruct(model, lr_img, patch_size, step, out_shape):
    lr_patches = extract_patches(lr_img, patch_size, step)
    preds = model.predict(lr_patches)

    patch_h, patch_w = patch_size
    img_h = (lr_img.shape[0] - patch_h) // step + 1
    img_w = (lr_img.shape[1] - patch_w) // step + 1

    result = np.zeros(out_shape)
    weight = np.zeros(out_shape)

    idx = 0
    for i in range(img_h):
        for j in range(img_w):
            patch = preds[idx].reshape(patch_h, patch_w)
            result[i*step:i*step+patch_h, j*step:j*step+patch_w] += patch
            weight[i*step:i*step+patch_h, j*step:j*step+patch_w] += 1
            idx += 1

    weight[weight == 0] = 1
    return result / weight

def random_forest_upscale(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img) / 255.0  # Normalize
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    hr_shape = (img.shape[0] * SCALE_FACTOR, img.shape[1] * SCALE_FACTOR)
    sr_channels = []

    for c in range(img.shape[2]):
        channel = img[:, :, c]
        hr_channel = transform.resize(channel, hr_shape, anti_aliasing=True)
        lr_channel = transform.resize(hr_channel, (hr_shape[0] // SCALE_FACTOR, hr_shape[1] // SCALE_FACTOR), anti_aliasing=True)
        lr_channel_up = transform.resize(lr_channel, hr_shape, anti_aliasing=True)

        X = extract_patches(lr_channel_up, PATCH_SIZE, STEP)
        y = extract_patches(hr_channel, PATCH_SIZE, STEP)

        if X.shape[0] > SAMPLE_PATCHES:
            idx = np.random.choice(X.shape[0], SAMPLE_PATCHES, replace=False)
            X = X[idx]
            y = y[idx]

        rf_model = train_rf(X, y)
        sr = predict_and_reconstruct(rf_model, lr_channel_up, PATCH_SIZE, STEP, hr_shape)
        sr_channels.append(sr)

    sr_image = np.stack(sr_channels, axis=-1)
    sr_image = np.clip(sr_image * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(sr_image)
