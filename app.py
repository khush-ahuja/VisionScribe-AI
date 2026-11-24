"""Streamlit demo: upload an image, get a caption (optionally read aloud).

    streamlit run app.py
"""
import io

import streamlit as st
import torch
from PIL import Image

from caption import caption_image
from evaluate import load_run

RUN_DIR = "runs/densenet121"


@st.cache_resource
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_run(RUN_DIR, device)
    return model, vocab, device


st.set_page_config(page_title="VisionScribe AI", page_icon="🖼️")
st.title("VisionScribe AI")
st.caption("Image captioning with a DenseNet-121 encoder and LSTM decoder, trained on Flickr8k.")

upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
beam = st.slider("Beam size", 1, 5, 3)
speak = st.checkbox("Read caption aloud")

if upload:
    image = Image.open(upload)
    st.image(image, use_container_width=True)
    model, vocab, device = get_model()
    with st.spinner("Generating caption..."):
        text = caption_image(model, vocab, image, device, beam)
    st.subheader(text.capitalize() + ".")
    if speak:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text).write_to_fp(buf)
        st.audio(buf.getvalue(), format="audio/mp3")
