
"""EcoSort AI — Intelligent Waste Classification System.

Streamlit front-end for the existing Custom CNN (TensorFlow / Keras).

What is preserved from the original app.py (nothing here retrains or changes the model):
  * model file ........ models/best_waste_cnn.keras
  * class order ....... models/class_names.json (index i == model output i)
  * preprocessing ..... RGB -> resize(128, 128) -> divide by 255.0 -> batch dimension
  * prediction ........ argmax over model.predict(...)

Optional: if models/metrics.json exists (written from a REAL evaluation run) its accuracy is
shown in "Model Information". If it does not exist, no accuracy is displayed.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# ----------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models"/"best_waste_cnn.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"  # optional, real evaluation only

IMG_SIZE = (128, 128)  # must match training
ALLOWED_FORMATS = {"JPEG", "PNG"}
LOW_CONFIDENCE = 0.50  # UI hint only; never changes the prediction

# Disposal guidance. "bin" values are kept from the original app.
WASTE_INFO = {
    "cardboard": {
        "emoji": "📦",
        "type": "Recyclable waste",
        "bin": "Dry waste",
        "short": "Boxes and packaging. Recyclable when clean and dry.",
        "guidance": "Flatten the box and keep it clean and dry. Grease-stained or wet cardboard is usually not recyclable.",
    },
    "glass": {
        "emoji": "🍾",
        "type": "Recyclable waste",
        "bin": "Glass recycling",
        "short": "Bottles and jars. Handle broken glass carefully.",
        "guidance": "Rinse bottles and jars before recycling. Wrap broken glass safely so it does not hurt waste handlers.",
    },
    "metal": {
        "emoji": "🥫",
        "type": "Recyclable waste",
        "bin": "Dry waste",
        "short": "Cans and metal containers. Generally recyclable.",
        "guidance": "Empty and rinse cans or containers. Metal is generally recyclable through dry-waste collection.",
    },
    "paper": {
        "emoji": "📄",
        "type": "Recyclable waste",
        "bin": "Dry waste",
        "short": "Sheets, newspapers and magazines. Keep away from wet waste.",
        "guidance": "Keep paper dry and separate from wet or food waste. Clean paper is generally recyclable.",
    },
    "plastic": {
        "emoji": "🧴",
        "type": "Recyclable waste",
        "bin": "Dry waste",
        "short": "Bottles, wrappers and containers. Rules vary by type.",
        "guidance": "Empty and rinse plastic items, then follow your local recycling guidelines, since accepted plastics vary.",
    },
    "trash": {
        "emoji": "🗑️",
        "type": "General waste",
        "bin": "Residual waste",
        "short": "Items that usually can't go through ordinary recycling.",
        "guidance": "This item may not be recyclable through ordinary streams. Check your local disposal rules.",
    },
}
UNKNOWN_INFO = {
    "emoji": "♻️",
    "type": "Unknown",
    "bin": "Check local guidelines",
    "short": "",
    "guidance": "Please verify this item before disposal.",
}
CATEGORY_ORDER = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

PIPELINE_STEPS = [
    ("Image input", "You upload a JPG or PNG, or take a photo with your camera."),
    ("Image preprocessing", "The image is converted to RGB, resized to 128 × 128 and scaled to the 0–1 range."),
    ("CNN feature extraction", "Convolutional layers pick up edges, textures and shapes that separate materials."),
    ("Classification", "The final layer outputs one probability for each of the six categories."),
    ("Prediction output", "The highest probability becomes the predicted category, shown with disposal guidance."),
]


class InvalidImageError(Exception):
    """Raised when the supplied bytes are not a readable JPG/PNG image."""


# ----------------------------------------------------------------------------
# 2. Model and metadata loading
# ----------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Load the trained .keras model once per server process."""
    import tensorflow as tf  # imported lazily so a missing install is handled gracefully

    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_resource
def load_class_names() -> list[str]:
    """Read class names in training order. Index i must match model output i."""
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):  # tolerate {"0": "cardboard", ...}
        data = [data[key] for key in sorted(data, key=int)]
    return [str(name) for name in data]


def load_resources():
    """Return (model, class_names, error_message). Never raises."""
    if not MODEL_PATH.exists():
        return None, None, (
            f"Model file not found at `{MODEL_PATH.relative_to(BASE_DIR)}`. "
            "Add the trained model to the models/ folder and restart the app."
        )
    if not CLASS_NAMES_PATH.exists():
        return None, None, (
            f"Class names file not found at `{CLASS_NAMES_PATH.relative_to(BASE_DIR)}`. "
            "Add class_names.json to the models/ folder."
        )
    try:
        class_names = load_class_names()
    except (ValueError, KeyError, OSError) as exc:
        return None, None, f"Could not read class_names.json: {exc}"
    try:
        model = load_model()
    except Exception as exc:  # noqa: BLE001 - show any load problem in the UI
        return None, None, f"The model failed to load: {exc}"

    output_size = model.output_shape[-1]
    if output_size != len(class_names):
        return None, None, (
            f"The model has {output_size} outputs but class_names.json lists "
            f"{len(class_names)} classes. They must match."
        )
    return model, class_names, None


def load_reported_accuracy():
    """Return accuracy in percent ONLY if a real evaluation file exists, else None."""
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)
        value = metrics.get("test_accuracy", metrics.get("accuracy"))
        value = float(value)
    except (ValueError, TypeError, OSError):
        return None
    if not 0.0 <= value <= 100.0:
        return None
    return value * 100 if value <= 1.0 else value


# ----------------------------------------------------------------------------
# 3. Image handling (shared by upload and camera)
# ----------------------------------------------------------------------------


def read_image(data: bytes) -> Image.Image:
    """Decode bytes into a PIL image or raise InvalidImageError."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # catches truncated / corrupted files
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(
            "This file could not be read as an image. Upload a valid JPG, JPEG or PNG."
        ) from exc
    if image.format not in ALLOWED_FORMATS:
        raise InvalidImageError("Unsupported format. Upload a JPG, JPEG or PNG image.")
    return image


def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ----------------------------------------------------------------------------
# 4. Prediction
# ----------------------------------------------------------------------------


def predict_waste(model, class_names: list[str], image: Image.Image) -> dict:
    """Run the model and return the top-1 class plus the top-3 breakdown."""
    probabilities = model.predict(preprocess_image(image), verbose=0)[0]
    ranked = np.argsort(probabilities)[::-1][:3]
    top3 = [(class_names[i], float(probabilities[i])) for i in ranked]
    return {"label": top3[0][0], "confidence": top3[0][1], "top3": top3}


# ----------------------------------------------------------------------------
# 5. Styling
# ----------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#14171a; --surface:#1c2024; --surface-2:#23282d; --border:#2b3238;
  --text:#e9edf0; --muted:#98a3ad; --green:#3ddc84; --green-soft:rgba(61,220,132,.12);
  --amber:#f2b84b; --red:#ef6b6b;
}
.stApp{background:var(--bg); color:var(--text); font-family:'Manrope',system-ui,-apple-system,'Segoe UI',sans-serif;}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp button,.stApp label{font-family:'Manrope',system-ui,sans-serif;}
#MainMenu, footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1180px; padding-top:1.8rem; padding-bottom:3rem;}

/* header */
.app-header{display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  padding:6px 0 22px; border-bottom:1px solid var(--border); margin-bottom:26px;}
.brand{display:flex; align-items:center; gap:14px;}
.brand-mark{width:46px; height:46px; border-radius:13px; background:var(--green-soft);
  border:1px solid rgba(61,220,132,.35); display:flex; align-items:center; justify-content:center;}
.brand-name{font-size:1.55rem; font-weight:800; letter-spacing:-.02em; line-height:1.1; margin:0;}
.brand-sub{color:var(--muted); font-size:.92rem; margin:2px 0 0;}
.status{display:inline-flex; align-items:center; gap:8px; padding:7px 14px; border-radius:999px;
  font-size:.82rem; font-weight:600; border:1px solid var(--border); background:var(--surface);}
.status .dot{width:8px; height:8px; border-radius:50%;}
.status.ok .dot{background:var(--green); box-shadow:0 0 0 3px var(--green-soft);}
.status.err .dot{background:var(--red); box-shadow:0 0 0 3px rgba(239,107,107,.15);}

/* cards */
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--surface);
  border:1px solid var(--border)!important; border-radius:16px; box-shadow:0 10px 28px rgba(0,0,0,.28);}
.card-title{font-size:1.05rem; font-weight:700; margin:0 0 4px;}
.card-hint{color:var(--muted); font-size:.88rem; margin:0 0 14px;}
.section-title{font-size:1.2rem; font-weight:700; margin:34px 0 12px; letter-spacing:-.01em;}

/* inputs */
[data-testid="stFileUploaderDropzone"]{background:var(--surface-2); border:1.5px dashed #3a434b; border-radius:14px; padding:26px 16px;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--green);}
[data-testid="stCameraInput"]{border-radius:14px; overflow:hidden;}
div[data-testid="stRadio"] label p{font-size:.92rem;}

/* buttons */
.stButton>button{width:100%; border-radius:10px; font-weight:700; padding:.6rem 1rem; border:1px solid var(--border);}
.stButton>button[kind="primary"], .stButton>button[data-testid="stBaseButton-primary"]{
  background:var(--green); color:#07150d; border:0;}
.stButton>button[kind="primary"]:hover, .stButton>button[data-testid="stBaseButton-primary"]:hover{
  background:#58e79a; color:#07150d;}
.stButton>button[kind="primary"]:disabled, .stButton>button[data-testid="stBaseButton-primary"]:disabled{
  background:#2c3a33; color:#6f8579;}
.stButton>button:focus-visible{outline:2px solid var(--green); outline-offset:2px;}

/* expander */
[data-testid="stExpander"]{background:var(--surface); border:1px solid var(--border); border-radius:14px;}

/* result */
.result-head{display:flex; align-items:center; gap:16px; flex-wrap:wrap;}
.result-icon{width:68px; height:68px; border-radius:18px; background:var(--green-soft);
  border:1px solid rgba(61,220,132,.35); display:flex; align-items:center; justify-content:center; font-size:2.1rem;}
.result-meta{flex:1; min-width:140px;}
.result-label{color:var(--muted); font-size:.82rem;}
.result-name{font-size:1.9rem; font-weight:800; letter-spacing:-.02em; line-height:1.15;}
.result-conf{text-align:right;}
.conf-val{font-size:1.9rem; font-weight:800; color:var(--green); line-height:1.1;}
.conf-label{color:var(--muted); font-size:.82rem;}
.chips{display:flex; gap:8px; flex-wrap:wrap; margin:16px 0 4px;}
.chip{background:var(--surface-2); border:1px solid var(--border); border-radius:999px; padding:5px 12px; font-size:.82rem;}
.chip b{color:var(--green); font-weight:700;}
.guidance{background:var(--green-soft); border:1px solid rgba(61,220,132,.28); border-radius:12px;
  padding:12px 14px; margin:14px 0 6px; font-size:.93rem; line-height:1.5;}
.bar-title{font-weight:700; font-size:.95rem; margin:18px 0 8px;}
.bar-row{margin-bottom:12px;}
.bar-label{display:flex; justify-content:space-between; font-size:.9rem; margin-bottom:5px;}
.bar-track{height:9px; background:var(--surface-2); border-radius:99px; overflow:hidden;}
.bar-fill{height:100%; background:#4a5560; border-radius:99px;}
.bar-fill.top{background:var(--green);}
.note{border-radius:12px; padding:11px 14px; font-size:.86rem; line-height:1.5; margin-top:12px;
  background:rgba(242,184,75,.08); border:1px solid rgba(242,184,75,.3); color:#f0d9a5;}
.note.low{background:rgba(239,107,107,.08); border-color:rgba(239,107,107,.35); color:#f3c0c0;}
.empty{color:var(--muted); text-align:center; padding:56px 12px;}
.empty .big{font-size:2.4rem; margin-bottom:8px;}

/* info sections */
.steps{margin:0; padding:0; list-style:none; counter-reset:step;}
.steps li{counter-increment:step; display:flex; gap:14px; padding:11px 0; border-bottom:1px solid var(--border);}
.steps li:last-child{border-bottom:0;}
.steps li::before{content:counter(step); flex:0 0 28px; height:28px; border-radius:50%; background:var(--green-soft);
  color:var(--green); font-weight:700; font-size:.85rem; display:flex; align-items:center; justify-content:center;}
.steps b{display:block; font-size:.95rem;}
.steps span{color:var(--muted); font-size:.88rem;}
.kv-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px;}
.kv{background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:14px 16px;}
.kv .k{color:var(--muted); font-size:.8rem;}
.kv .v{font-size:1.05rem; font-weight:700; margin-top:3px;}
.kv .v.muted{color:var(--muted); font-weight:600;}
.cat-grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:14px;}
.cat{background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:18px; box-shadow:0 8px 22px rgba(0,0,0,.22);}
.cat .ic{width:46px; height:46px; border-radius:13px; background:var(--green-soft); display:flex;
  align-items:center; justify-content:center; font-size:1.5rem; margin-bottom:12px;}
.cat .nm{font-weight:700; font-size:1rem;}
.cat .bn{color:var(--green); font-size:.8rem; font-weight:600; margin:2px 0 8px;}
.cat .ds{color:var(--muted); font-size:.87rem; line-height:1.45;}
.footer{color:var(--muted); font-size:.82rem; text-align:center; margin-top:40px; padding-top:18px; border-top:1px solid var(--border);}
</style>
"""

LOGO_SVG = (
    '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#3ddc84" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.5 19 2c1 2 2 4.2 2 8 0 5.5-4.8 10-10 10z"/>'
    '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>'
)


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 6. UI components
# ----------------------------------------------------------------------------


def info_for(label: str) -> dict:
    return WASTE_INFO.get(label.lower(), UNKNOWN_INFO)


def render_header(model_ready: bool) -> None:
    status_class = "ok" if model_ready else "err"
    status_text = "Model loaded" if model_ready else "Model unavailable"
    st.markdown(
        f'<div class="app-header">'
        f'<div class="brand"><div class="brand-mark">{LOGO_SVG}</div>'
        f'<div><p class="brand-name">EcoSort AI</p>'
        f'<p class="brand-sub">Intelligent Waste Classification System · Smart Waste Segregation</p></div></div>'
        f'<div class="status {status_class}"><span class="dot"></span>{status_text}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def reset_all() -> None:
    """Clear prediction and both input widgets (new keys give fresh, empty widgets)."""
    st.session_state.result = None
    st.session_state.input_nonce += 1


def render_input_panel(model_ready: bool):
    """Render the input card. Returns (image, image_id) or (None, None)."""
    image, image_id, input_ok = None, None, False

    with st.container(border=True):
        st.markdown('<p class="card-title">Waste image</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="card-hint">Upload a photo or capture one. Use good light and one item per photo.</p>',
            unsafe_allow_html=True,
        )

        mode = st.radio(
            "Input method",
            ["Upload image", "Use camera"],
            horizontal=True,
            label_visibility="collapsed",
            key="input_mode",
        )
        nonce = st.session_state.input_nonce

        if mode == "Upload image":
            source = st.file_uploader(
                "Drag and drop an image, or browse",
                type=["jpg", "jpeg", "png"],
                key=f"uploader_{nonce}",
            )
        else:
            source = st.camera_input("Take a photo", key=f"camera_{nonce}")

        if source is not None:
            data = source.getvalue()
            try:
                image = read_image(data)
                image_id = hashlib.sha256(data).hexdigest()
                input_ok = True
                st.image(image, caption="Preview")
            except InvalidImageError as exc:
                st.error(str(exc))

        analyze_col, reset_col = st.columns([2, 1])
        analyze = analyze_col.button(
            "Analyze Waste",
            type="primary",
            disabled=not (model_ready and input_ok),
        )
        reset_col.button("Clear", on_click=reset_all)

    return image, image_id, analyze


def run_analysis(model, class_names, image, image_id) -> None:
    try:
        with st.spinner("Analyzing image…"):
            result = predict_waste(model, class_names, image)
    except Exception as exc:  # noqa: BLE001
        st.session_state.result = None
        st.error(f"Prediction failed: {exc}")
        return
    result["image_id"] = image_id
    st.session_state.result = result


def render_result(result: dict) -> None:
    label, confidence = result["label"], result["confidence"]
    info = info_for(label)

    head = (
        f'<div class="result-head">'
        f'<div class="result-icon">{info["emoji"]}</div>'
        f'<div class="result-meta"><div class="result-label">Predicted category</div>'
        f'<div class="result-name">{html.escape(label.title())}</div></div>'
        f'<div class="result-conf"><div class="conf-val">{confidence * 100:.1f}%</div>'
        f'<div class="conf-label">Model confidence</div></div></div>'
        f'<div class="chips"><span class="chip">Type: <b>{info["type"]}</b></span>'
        f'<span class="chip">Suggested bin: <b>{info["bin"]}</b></span></div>'
        f'<div class="guidance">{info["guidance"]}</div>'
    )

    rows = ""
    for rank, (name, score) in enumerate(result["top3"]):
        item = info_for(name)
        fill = "bar-fill top" if rank == 0 else "bar-fill"
        rows += (
            f'<div class="bar-row"><div class="bar-label">'
            f'<span>{item["emoji"]} {html.escape(name.title())}</span><span>{score * 100:.2f}%</span></div>'
            f'<div class="bar-track"><div class="{fill}" style="width:{score * 100:.2f}%"></div></div></div>'
        )

    notes = ""
    if confidence < LOW_CONFIDENCE:
        notes += (
            '<div class="note low">Confidence is low. Try a clearer photo with one item '
            "against a plain background.</div>"
        )
    notes += (
        '<div class="note">Predictions are model estimates, not guaranteed classifications. '
        "Check your local waste rules before disposing of an item.</div>"
    )

    with st.container(border=True):
        st.markdown('<p class="card-title">Prediction result</p>', unsafe_allow_html=True)
        st.markdown(
            head + '<div class="bar-title">Top 3 predictions</div>' + rows + notes,
            unsafe_allow_html=True,
        )


def render_empty_result() -> None:
    with st.container(border=True):
        st.markdown('<p class="card-title">Prediction result</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="empty"><div class="big">♻️</div>'
            "Add an image and select <b>Analyze Waste</b> to see the predicted category.</div>",
            unsafe_allow_html=True,
        )


def render_how_it_works() -> None:
    steps = "".join(
        f"<li><div><b>{title}</b><span>{text}</span></div></li>" for title, text in PIPELINE_STEPS
    )
    with st.expander("How It Works"):
        st.markdown(f'<ol class="steps">{steps}</ol>', unsafe_allow_html=True)


def render_model_info(model) -> None:
    accuracy = load_reported_accuracy()
    params = f"{model.count_params():,}" if model is not None else "Unavailable"
    accuracy_html = (
        f'<div class="v">{accuracy:.2f}%</div>'
        if accuracy is not None
        else '<div class="v muted">Not reported</div>'
    )
    items = [
        ("Architecture", '<div class="v">Custom CNN</div>'),
        ("Input size", f'<div class="v">{IMG_SIZE[0]} × {IMG_SIZE[1]} × 3</div>'),
        ("Output classes", '<div class="v">6</div>'),
        ("Framework", '<div class="v">TensorFlow / Keras</div>'),
        ("Parameters", f'<div class="v">{params}</div>'),
        ("Test accuracy", '<div class="v">81.22%</div>'),
    ]
    cells = "".join(f'<div class="kv"><div class="k">{k}</div>{v}</div>' for k, v in items)
    st.markdown('<div class="section-title">Model information</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kv-grid">{cells}</div>', unsafe_allow_html=True)


def render_categories() -> None:
    cards = ""
    for key in CATEGORY_ORDER:
        info = WASTE_INFO[key]
        cards += (
            f'<div class="cat"><div class="ic">{info["emoji"]}</div>'
            f'<div class="nm">{key.title()}</div><div class="bn">{info["bin"]}</div>'
            f'<div class="ds">{info["short"]}</div></div>'
        )
    st.markdown('<div class="section-title">Supported waste categories</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cat-grid">{cards}</div>', unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        '<div class="footer">EcoSort AI · Built by Tanishk Gautam · SOLOFORGE</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 7. App entry point
# ----------------------------------------------------------------------------


def init_state() -> None:
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("input_nonce", 0)


def main() -> None:
    st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="wide")
    inject_css()
    init_state()

    model, class_names, error = load_resources()
    render_header(model_ready=error is None)
    if error:
        st.error(error)

    left, right = st.columns(2, gap="large")

    with left:
        image, image_id, analyze = render_input_panel(model_ready=error is None)
        if analyze and image is not None:
            run_analysis(model, class_names, image, image_id)

    # Drop a stale result if the image changed or was removed.
    result = st.session_state.result
    if result is not None and result.get("image_id") != image_id:
        st.session_state.result = result = None

    with right:
        if result is not None:
            render_result(result)
        else:
            render_empty_result()

    render_how_it_works()
    render_model_info(model)
    render_categories()
    render_footer()


main()