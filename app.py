import gradio as gr
from PIL import Image
import os

# Existing imports
from models.lancros_interpolation import upsample_lancros
from models.fourier_interpolation import fourier_upscale
from models.autoencoder import autoencoder_upscale
from models.espcn import espcn_upscale
from models.sr_gan import srgan_upscale

# ✅ New import for Random Forest Super Resolution
from models.random_forest_sr import random_forest_upscale

lancros_page = gr.Interface(
    fn=upsample_lancros,
    inputs=[gr.Image(label="Low Resolution Image"),
            gr.Slider(2, 6, step=1, value=2, label="Upscaling Factor"),],
    outputs=gr.Image(type="pil", label="High Resolution Images"),
    title="Lancros Upsampling"
)

fourier_page = gr.Interface(
    fn=fourier_upscale,
    inputs=[gr.Image(label="Low Resolution Image"),
            gr.Slider(2, 6, step=1, value=2, label="Upscaling Factor"),],
    outputs=gr.Image(type="pil", label="High Resolution Images"),
    title="Fourier Upsampling"
)

autoencoder_page = gr.Interface(
    fn=autoencoder_upscale,
    inputs=[gr.Image(label="Low Resolution Image")],
    outputs=gr.Image(type="pil", label="High Resolution Images"),
    title="Autoencoder based Super Resolution"
)

espcn_page = gr.Interface(
    fn=espcn_upscale,
    inputs=[gr.Image(label="Low Resolution Image")],
    outputs=gr.Image(type="pil", label="High Resolution Images"),
    title="ESPCN based Super Resolution"
)

srgan_page = gr.Interface(
    fn=srgan_upscale,
    inputs=[gr.Image(label="Low Resolution Image")],
    outputs=gr.Image(type="pil", label="High Resolution Images"),
    title="GAN based Super Resolution"
)

# ✅ Random Forest page
random_forest_page = gr.Interface(
    fn=random_forest_upscale,
    inputs=[gr.Image(label="Low Resolution Image")],
    outputs=gr.Image(type="pil", label="High Resolution Images"),
    title="Random Forest based Super Resolution"
)

demo = gr.TabbedInterface(
    [lancros_page, fourier_page, autoencoder_page, srgan_page, random_forest_page],
    ["Lancros Interpolation", "Fourier Interpolation", "Autoencoder based Super Resolution", "GAN based Super Resolution", "Random Forest based Super Resolution"],
    title="Image Super Resolution"
)

if __name__ == "__main__":
    demo.launch(server_name="172.31.94.47")
