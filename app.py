import gradio as gr
from PIL import Image
import os

from models.lancros_interpolation import upsample_lancros
from models.fourier_interpolation import fourier_upscale
from models.autoencoder import autoencoder_upscale
from models.espcn import espcn_upscale
from models.sr_gan import srgan_upscale

from models.random_forest_sr import random_forest_upscale

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
        ["sample_images/0003.png"],
        ["sample_images/0055.png"]
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

demo = gr.TabbedInterface(
    [srgan_page,lancros_page, fourier_page, autoencoder_page, random_forest_page, espcn_page],
    ["GAN based Super Resolution","Lancros Interpolation", "Fourier Interpolation", "Autoencoder based Super Resolution", "Random Forest based Super Resolution","ESPCN based Super Resolution"],
    title="Image Super Resolution"
)

if __name__ == "__main__":
    demo.launch(debug=True)
