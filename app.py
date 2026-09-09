"""Browser interface for the deepfake CNN detector."""

from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image
import streamlit as st

CODE_DIR = Path(__file__).resolve().parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from model_utils import DEFAULT_MODEL_PATH, load_model, predict_image


st.set_page_config(page_title="Deepfake Image Detector", page_icon="🔍", layout="centered")
st.title("Deepfake Image Detector")
st.write("Upload a face image and the trained custom CNN will classify it as real or fake")


@st.cache_resource
def cached_model():
    return load_model(DEFAULT_MODEL_PATH)


try:
    model = cached_model()
except Exception as error:
    st.error(f"The model could not be loaded: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=("jpg", "jpeg", "png", "bmp", "webp"),
)

if uploaded_file is not None:
    try:
        with Image.open(uploaded_file) as opened_image:
            display_image = opened_image.convert("RGB")
        st.image(display_image, caption="Uploaded image", use_container_width=True)
        result = predict_image(model, display_image)

        label = str(result["label"]).upper()
        confidence = float(result["confidence"])
        if label == "FAKE":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        first, second = st.columns(2)
        first.metric("Confidence", f"{confidence * 100:.2f}%")
        second.metric("Fake probability", f"{float(result['probability_fake']) * 100:.2f}%")
        st.progress(confidence)
    except Exception as error:
        st.error(f"The image could not be processed: {error}")

st.warning(
    "Research prototype only. The training data contains composition differences "
    "between real and fake images so predictions should not be treated as proof."
)
