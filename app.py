import gradio as gr
from PIL import Image
import os
import numpy as np
import cv2

# Existing imports
from models.lancros_interpolation import upsample_lancros
from models.fourier_interpolation import fourier_upscale
from models.autoencoder import autoencoder_upscale
from models.espcn import espcn_upscale
from models.sr_gan import srgan_upscale
from models.cnn import srcnn_upscale

# ✅ New import for Random Forest Super Resolution
from models.random_forest_sr import random_forest_upscale

# ✅ EDI (Edge Directed Interpolation) method
from PIL import Image
import numpy as np

from PIL import Image
import numpy as np
from models.rcan import rcan_upscale

def edge_directed_interpolation(lr_img_pil, scale=2):
    # Ensure input is a PIL Image

    if isinstance(lr_img_pil, np.ndarray):
        lr_img_pil = Image.fromarray(lr_img_pil)

    # Convert to RGB
    lr_img_rgb = lr_img_pil.convert("RGB")
    lr_img = np.array(lr_img_rgb)

    h, w, c = lr_img.shape
    hr_h, hr_w = h * scale, w * scale
    hr_img = np.zeros((hr_h, hr_w, c), dtype=np.uint8)

    # Copy original pixels to even positions
    for i in range(h):
        for j in range(w):
            hr_img[i * scale, j * scale] = lr_img[i, j]

    # Interpolate diagonal pixels per channel
    for i in range(0, hr_h, scale):
        for j in range(0, hr_w, scale):
            if i + scale < hr_h and j + scale < hr_w:
                for ch in range(c):
                    p1 = hr_img[i, j, ch]
                    p2 = hr_img[i, j + scale, ch]
                    p3 = hr_img[i + scale, j, ch]
                    p4 = hr_img[i + scale, j + scale, ch]

                    d1 = abs(int(p1) - int(p4))
                    d2 = abs(int(p2) - int(p3))

                    interp = (int(p1) + int(p4)) // 2 if d1 < d2 else (int(p2) + int(p3)) // 2
                    hr_img[i + scale // 2, j + scale // 2, ch] = interp

    # Fill remaining zero pixels per channel
    for i in range(hr_h):
        for j in range(hr_w):
            for ch in range(c):
                if hr_img[i, j, ch] == 0:
                    neighbors = []
                    if i - 1 >= 0:
                        neighbors.append(hr_img[i - 1, j, ch])
                    if i + 1 < hr_h:
                        neighbors.append(hr_img[i + 1, j, ch])
                    if j - 1 >= 0:
                        neighbors.append(hr_img[i, j - 1, ch])
                    if j + 1 < hr_w:
                        neighbors.append(hr_img[i, j + 1, ch])
                    if neighbors:
                        hr_img[i, j, ch] = int(np.mean(neighbors))

    return Image.fromarray(hr_img)



# === Interfaces === #
lancros_page = gr.Interface(
    fn=upsample_lancros,
    inputs=[gr.Image(label="Low Resolution Image"),
            gr.Slider(2, 6, step=1, value=2, label="Upscaling Factor"),],
    outputs=gr.Image(type="pil", label="High Resolution Images"),
    title="Lancros Upsampling",
    examples=[
        ["sample_images/0001.png"],
        ["sample_images/0172.png"]
    ]
)

fourier_page = gr.Interface(
    fn=fourier_upscale,
    inputs=[gr.Image(label="Low Resolution Image"),
            gr.Slider(2, 6, step=1, value=2, label="Upscaling Factor"),],
    outputs=gr.Image(type="pil", label="High Resolution Images"),
    title="Fourier Upsampling",
    examples=[
        ["sample_images/0004.png"],
        ["sample_images/0012.png"]
    ]
)

autoencoder_page = gr.Interface(
    fn=autoencoder_upscale,
    inputs=[gr.Image(label="Low Resolution Image")],
    outputs=gr.Image(type="pil", label="High Resolution Images"),
    title="Autoencoder based Super Resolution",
    examples=[
        ["sample_images/0019.png"],
        ["sample_images/0064.png"]
    ]
)

espcn_page = gr.Interface(
    fn=espcn_upscale,
    inputs=[gr.Image(label="Low Resolution Image")],
    outputs=gr.Image(type="pil", label="High Resolution Images"),
    title="ESPCN based Super Resolution",
    examples=[
        ["sample_images/0024.png"],
        ["sample_images/0068.png"]
    ]
)

srgan_page = gr.Interface(
    fn=srgan_upscale,
    inputs=[gr.Image(label="Low Resolution Image")],
    outputs=gr.Image(type="pil", label="High Resolution Images"),
    title="GAN based Super Resolution",
    examples=[
        ["sample_images/0055.png"],
        ["sample_images/0003.png"]
    ]
)

random_forest_page = gr.Interface(
    fn=random_forest_upscale,
    inputs=[gr.Image(label="Low Resolution Image")],
    outputs=gr.Image(type="pil", label="High Resolution Images"),
    title="Random Forest based Super Resolution",
    examples=[
        ["sample_images/0097.png"],
        ["sample_images/0086.png"]
    ]
)

# ✅ EDI Page
edi_page = gr.Interface(
    fn=edge_directed_interpolation,
    inputs=[gr.Image(label="Low Resolution Image"), gr.Slider(2, 4, step=1, value=2, label="Upscaling Factor")],
    outputs=gr.Image(type="pil", label="High Resolution Image"),
    title="Edge Directed Interpolation",
    examples=[
        ["sample_images/0002.png"],
        ["sample_images/0006.png"]
    ]
)

rcan_page = gr.Interface(
    fn=rcan_upscale,
    inputs=[gr.Image(label="Low Resolution Image")],
    outputs=gr.Image(type="pil", label="High Resolution Image"),
    title="RCAN based Super Resolution",
    examples=[
        ["sample_images/0007.png"],
        ["sample_images/0010.png"]
    ]
)

srcnn_page = gr.Interface(
    fn=srcnn_upscale,
    inputs=[gr.Image(label="Low Resolution Image")],
    outputs=gr.Image(type="pil", label="High Resolution Image"),
    title="SRCNN based Super Resolution",
    examples=[
        ["sample_images/0007.png"],
        ["sample_images/0010.png"]
    ]
)

# Tabs setup
demo = gr.TabbedInterface(
    [srgan_page, lancros_page,espcn_page, random_forest_page, edi_page, rcan_page,srcnn_page, autoencoder_page,fourier_page],
    ["GAN based Super Resolution", "Lancros Interpolation", 
     "EspCN Super Resolution", "Random Forest based Super Resolution", "Edge Directed Interpolation", "RCAN Super Resolution","SRCNN Super Resolution","Autoencoder based Super Resolution", "Fourier Interpolation"],
    title="Image Super Resolution"
)

if __name__ == "__main__":
    demo.launch(debug=True)