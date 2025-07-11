import streamlit as st
from streamlit_image_comparison import image_comparison
from pathlib import Path
import torch
import cv2
from PIL import Image
import numpy as np
from omegaconf import OmegaConf


st.set_page_config(page_title="IR Demo", layout="wide")
st.title("Interactive Image Reconstruction")


@st.cache_resource
def load_model(checkpoint: Path):
    """Load a traced model (gpu)."""
    model = torch.jit.load(checkpoint)
    return model


@st.cache_data
def load_config():
    config = OmegaConf.load(Path(__file__).parent / "config.yaml")
    return config


def reset_state():
    st.session_state.done = False
    st.session_state.img1 = None
    st.session_state.img2 = None
    
    
def load_image(uploaded_file) -> torch.Tensor:
    # return torch.from_numpy(
    #     cv2.imread(str(p), -1)[:, :, 0]
    # )  # in [0, 255]; pixel; HxW; uint8
    return torch.from_numpy(
        np.asarray(Image.open(uploaded_file))[:, :, 0]
    )  # uint8; HxW

    
def to_Kelvin(img: torch.Tensor, min_temp, max_temp) -> torch.Tensor:
    return ((max_temp - min_temp) * (img / 255.0)) + min_temp  # in Kelvin


def tone_mapping(
    x: torch.Tensor,  # HxW
    min: float,
    max: float,
    colormap: int = cv2.COLORMAP_INFERNO,
) -> Image.Image:
    assert x.ndim == 2  # HxW
    x = (x.clamp(min, max) - min) / (max - min)  # [0.0, 1.0]
    x = (255 * x).to(dtype=torch.uint8).numpy()  # [0, 255]; uint8
    x = cv2.applyColorMap(x, colormap)  # HxWxC
    return Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))  # HxWxC
    
    
config = load_config()
model = load_model(config.checkpoint)

uploaded_file = st.file_uploader(
    "Upload the AOS image",
    type=["png"],
    on_change=reset_state
)

# Batch1 subset stats.
aos_mean = 291.3995303831239
aos_std = 9.99302877546057

if uploaded_file:
    if not st.session_state.done:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_temp = st.number_input("Min. AOS temperature", value=265.48, format="%.2f")
        
        with col2:
            max_temp = st.number_input("Max. AOS temperature", value=367.52, format="%.2f")
        
        with col3:
            env_temp = torch.tensor([st.number_input(
                "Select the environment temperature",
                min_value=0,
                max_value=30,
                value=0,
                step=1,
            )]).to(torch.int64).cuda()
            
        left, center, right = st.columns([1, 2, 1])
        with center:
            if st.button("Process Image"):
                print(f"{min_temp=} {max_temp=} {env_temp=}")
                
                img1 = to_Kelvin(
                    load_image(uploaded_file),
                    min_temp,
                    max_temp,
                )  # HxW
                img1_normalized = (img1 - aos_mean) / aos_std
                
                with torch.inference_mode():
                    img2_normalized: torch.Tensor = model(
                        img1_normalized[None, None, :, :].cuda(),
                        env_temp,
                        None,
                        True,
                    )
                    img2 = aos_std * img2_normalized.cpu() + aos_mean  # 1x1xHxW
                
                img1 = tone_mapping(img1, min_temp, max_temp)
                img2 = tone_mapping(img2[0, 0], min_temp, max_temp)
                st.session_state.img1 = img1
                st.session_state.img2 = img2
                st.session_state.done = True
                st.rerun()
    else:
        image_comparison(
            img1=st.session_state.img1,
            img2=st.session_state.img2,
            label1="Original",
            label2="Corrected",
        )
else:
    st.info("Upload an image to get started.")
