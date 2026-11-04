import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_waste_cnn.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

IMG_SIZE = (128, 128)  # training ke time wala size
LOW_CONFIDENCE = 0.50

# emoji, bin, tip
WASTE_INFO = {
    "cardboard": ("📦", "Dry waste", "Box flat karke clean aur dry rakho. Geela ya greasy cardboard recycle nahi hota."),
    "glass": ("🍾", "Glass recycling", "Bottle/jar rinse karo. Toota glass safely wrap karke do."),
    "metal": ("🥫", "Dry waste", "Can ya container khali karke rinse karo. Metal generally recyclable hai."),
    "paper": ("📄", "Dry waste", "Paper ko dry rakho aur wet/food waste se alag rakho."),
    "plastic": ("🧴", "Dry waste", "Rinse karke local recycling rules follow karo, sab plastic accept nahi hota."),
    "trash": ("🗑️", "Residual waste", "Ye item shayad recycle na ho. Local disposal rules check karo."),
}
DEFAULT_INFO = ("♻️", "Check local guidelines", "Disposal se pehle item verify kar lo.")


# ---------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Model load ho raha hai...")
def load_model():
    import tensorflow as tf

    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_resource
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data[k] for k in sorted(data, key=int)]
    return [str(name) for name in data]


# ---------------------------------------------------------------
# Preprocess + predict (same as original, kuch change nahi)
# ---------------------------------------------------------------
def preprocess(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, class_names, image):
    probs = model.predict(preprocess(image), verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    return [(class_names[i], float(probs[i])) for i in top]


# ---------------------------------------------------------------
# App
# ---------------------------------------------------------------
st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="wide")

st.title("♻️ EcoSort AI")
st.caption("Intelligent Waste Classification System · Smart Waste Segregation")

# Files check
if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    st.error("`models/best_waste_cnn.keras` ya `models/class_names.json` nahi mili. Dono ko models/ folder mein rakho.")
    st.stop()

model = load_model()
class_names = load_class_names()

if model.output_shape[-1] != len(class_names):
    st.error(
        f"Model ke {model.output_shape[-1]} outputs hain, par class_names.json mein "
        f"{len(class_names)} classes hain. Dono match hone chahiye."
    )
    st.stop()

# Sidebar: model info
with st.sidebar:
    st.header("Model information")
    st.write("**Architecture:** Custom CNN")
    st.write(f"**Input size:** {IMG_SIZE[0]} × {IMG_SIZE[1]} × 3")
    st.write(f"**Output classes:** {len(class_names)}")
    st.write("**Framework:** TensorFlow / Keras")
    st.write(f"**Parameters:** {model.count_params():,}")
    st.write("**Test accuracy:** 81.22%")
    st.divider()
    st.caption("Built by Tanishk Gautam · SOLOFORGE")

left, right = st.columns(2, gap="large")

# ---- Left: input ----
with left:
    st.subheader("Waste image")
    st.caption("Photo upload karo ya camera se lo. Achhi light aur ek photo mein ek hi item.")

    tab_upload, tab_camera = st.tabs(["Upload image", "Use camera"])
    with tab_upload:
        uploaded = st.file_uploader("JPG / PNG chuno", type=["jpg", "jpeg", "png"])
    with tab_camera:
        captured = st.camera_input("Photo kheencho")

    source = uploaded if uploaded is not None else captured

    image = None
    if source is not None:
        try:
            image = Image.open(source)
            image.load()
            st.image(image, caption="Preview", use_container_width=True)
        except Exception:
            st.error("Ye file image ki tarah read nahi ho payi. Valid JPG/PNG upload karo.")
            image = None

    analyze = st.button("Analyze Waste", type="primary", disabled=image is None, use_container_width=True)

# ---- Right: result ----
with right:
    st.subheader("Prediction result")

    if analyze and image is not None:
        with st.spinner("Analyze ho raha hai..."):
            top3 = predict(model, class_names, image)

        label, confidence = top3[0]
        emoji, bin_name, tip = WASTE_INFO.get(label.lower(), DEFAULT_INFO)

        c1, c2 = st.columns(2)
        c1.metric("Predicted category", f"{emoji} {label.title()}")
        c2.metric("Confidence", f"{confidence * 100:.1f}%")

        st.success(f"Suggested bin: **{bin_name}**")
        st.info(tip)

        if confidence < LOW_CONFIDENCE:
            st.warning("Confidence kam hai. Plain background pe ek item ki saaf photo try karo.")

        st.markdown("**Top 3 predictions**")
        for name, score in top3:
            e = WASTE_INFO.get(name.lower(), DEFAULT_INFO)[0]
            st.write(f"{e} {name.title()} — {score * 100:.2f}%")
            st.progress(min(max(score, 0.0), 1.0))

        st.caption("Predictions model ke estimates hain, guaranteed nahi. Disposal se pehle local rules check karo.")
    else:
        st.info("Image add karo aur **Analyze Waste** dabao, predicted category yahan dikhegi.")

# ---- Bottom: info ----
st.divider()

with st.expander("How it works"):
    st.markdown(
        "1. **Image input**: JPG/PNG upload ya camera photo\n"
        "2. **Preprocessing**: RGB convert, 128×128 resize, 0–1 scale\n"
        "3. **CNN feature extraction**: edges, textures aur shapes pehchaane jaate hain\n"
        "4. **Classification**: 6 categories ke liye probabilities\n"
        "5. **Output**: sabse high probability wali category + disposal guidance"
    )

with st.expander("Supported waste categories"):
    cols = st.columns(3)
    for i, (name, (emoji, bin_name, tip)) in enumerate(WASTE_INFO.items()):
        with cols[i % 3]:
            st.markdown(f"**{emoji} {name.title()}**  \n{bin_name}  \n{tip}")