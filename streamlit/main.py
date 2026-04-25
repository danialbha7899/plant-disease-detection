import json
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = SCRIPT_DIR / "trained_model.keras"
CLASS_NAMES_PATH = SCRIPT_DIR / "class_names.json"
TRAIN_DIR = SCRIPT_DIR.parent / "train"

_GALLERY = (
    ("grape", "Grape", "Grape — leaf diseases including rot and blight are covered by the model."),
    ("strawberry", "Strawberry", "Strawberry — e.g. leaf scorch and healthy class."),
    ("tomato", "Tomato", "Tomato — the largest set of conditions in the dataset."),
)


def _find_image(base_name: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG"):
        p = SCRIPT_DIR / f"{base_name}{ext}"
        if p.is_file():
            return p
    return None


def _load_class_names() -> list[str]:
    if CLASS_NAMES_PATH.is_file():
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            return json.load(f)
    if TRAIN_DIR.is_dir():
        return sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
    raise FileNotFoundError("Need streamlit/class_names.json or a ../train folder.")


@st.cache_resource
def get_model() -> tf.keras.Model:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Place the model at: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def _format_label(raw: str) -> str:
    if "___" in raw:
        a, b = raw.split("___", 1)
        return f"{a.replace('_', ' ').strip()} — {b.replace('_', ' ').strip()}"
    return raw.replace("_", " ")


def _parse_class_label(raw: str) -> tuple[str, bool, str]:
    """(plant_pretty, is_healthy, problem_pretty). problem empty if healthy."""
    if "___" not in raw:
        return raw.replace("_", " ").strip(), False, "an unknown"
    plant, rest = raw.split("___", 1)
    plant_pretty = plant.replace("_", " ").strip()
    if rest == "healthy":
        return plant_pretty, True, ""
    problem_pretty = rest.replace("_", " ").strip()
    return plant_pretty, False, problem_pretty


def _prediction_narration(raw: str) -> str:
    plant, is_healthy, problem = _parse_class_label(raw)
    if is_healthy:
        return (
            f"The model predicted your leaf is a **{plant}** leaf and the leaf is **healthy**."
        )
    return (
        f"The model predicted your leaf is a **{plant}** leaf and the leaf is **unhealthy** with "
        f"**{problem}** problem."
    )


def model_prediction(test_image) -> int:
    model = get_model()
    img = (
        Image.open(BytesIO(test_image.getvalue()))
        .convert("RGB")
        .resize((128, 128), Image.BILINEAR)
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    return int(np.argmax(probs))


# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Plant health · Leaf check",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

class_name = _load_class_names()

# ------------------ THEME: dark gray background, white / light text ------------------
st.markdown(
    """
    <style>
    :root {
      --pd-bg: #1a1a1a;
      --pd-bg-elev: #262626;
      --pd-text: #f9fafb;
      --pd-text-muted: #d1d5db;
    }
    .stApp,
    [data-testid="stAppViewContainer"] {
      background: var(--pd-bg) !important;
      color: var(--pd-text) !important;
      color-scheme: dark;
    }
    [data-testid="stHeader"] {
      background: #121212 !important;
      border-bottom: 1px solid #2d2d2d;
    }
    [data-testid="stAppViewContainer"] > .main,
    section.main > div,
    .block-container,
    div[data-testid="stMainBlockContainer"] {
      background: transparent !important;
    }
    section[data-testid="stMain"] {
      color: var(--pd-text) !important;
    }
    .stApp p, .stApp li, .stApp label,
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown div,
    [data-testid="stVerticalBlock"] p,
    [data-testid="stVerticalBlock"] li {
      color: var(--pd-text-muted) !important;
    }
    .stMarkdown strong {
      color: #ffffff !important;
    }
    h1, h2, h3, h4, [data-testid="stHeader"] a {
      color: #ffffff !important;
    }
    [data-testid="stCaption"], .stCaption, small {
      color: #9ca3af !important;
    }
    /* Upload block: drag-and-drop only — hide "Browse files" (Base Web + modern internal button) */
    [data-testid="stFileUploader"] [data-baseweb="button"] {
      display: none !important;
    }
    [data-testid="stFileUploaderDropzone"]:has([data-testid="stFileUploaderDropzoneInstructions"])
      > span:has(> button) {
      display: none !important;
    }
    [data-testid="stFileUploader"] {
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 100% !important;
    }
    [data-testid="stFileUploader"] > section,
    [data-testid="stFileUploader"] section {
      background: transparent !important;
      border: none !important;
      box-shadow: none !important;
      border-radius: 0 !important;
      padding: 0 !important;
      display: flex !important;
      flex-direction: row;
      flex-wrap: wrap;
      justify-content: center !important;
      align-items: center;
      gap: 0.5rem;
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] label {
      color: #9ca3af !important;
    }
    [data-testid="stFileUploader"] small {
      flex-basis: 100%;
      text-align: center !important;
      margin-top: 0.35rem !important;
    }
    /* st.container(border=True, key="pd_leaf_upload") — dark card */
    [class*="-pd_leaf_upload"] {
      background: var(--pd-bg-elev) !important;
      border: 1px solid #3a3a3a !important;
      border-radius: 12px !important;
      box-shadow: 0 1px 0 rgba(255, 255, 255, 0.04) inset;
      padding: 0.85rem 0.9rem 1rem !important;
    }
    [class*="-pd_leaf_upload"] p {
      text-align: center;
    }
    /* Primary CTA (Analyze) */
    .stButton > button[kind="primary"] {
      background: linear-gradient(180deg, #16a34a, #15803d) !important;
      color: #f0fdf4 !important;
      border: none !important;
      border-radius: 999px !important;
      min-height: 2.75rem;
      font-weight: 600 !important;
      letter-spacing: 0.03em;
      box-shadow: 0 6px 24px rgba(22, 163, 74, 0.38) !important;
    }
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span {
      color: #f0fdf4 !important;
    }
    .stButton > button[kind="primary"]:hover {
      filter: brightness(1.1);
    }
    [data-testid="stImage"] img {
      border-radius: 12px !important;
      box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4) !important;
    }
    /* Dividers */
    hr {
      border-color: #3a3a3a !important;
    }
    /* st.info / generic alerts (not the green result card) */
    [data-baseweb="notification"] {
      background: #2a2a2a !important;
      color: #f3f4f6 !important;
    }
    [data-baseweb="notification"] a,
    div[data-baseweb="notification"] * {
      color: inherit !important;
    }
    /* Prediction result: st.success — glassy green card */
    [data-testid="stSuccess"] {
      background: linear-gradient(
        125deg,
        rgba(4, 47, 46, 0.75) 0%,
        rgba(15, 81, 50, 0.65) 45%,
        rgba(5, 46, 22, 0.7) 100%
      ) !important;
      border: 1px solid rgba(52, 211, 153, 0.45) !important;
      border-left: 4px solid #34d399 !important;
      border-radius: 16px !important;
      box-shadow:
        0 0 0 1px rgba(255, 255, 255, 0.06) inset,
        0 8px 32px rgba(0, 0, 0, 0.45),
        0 0 40px -8px rgba(16, 185, 129, 0.35) !important;
      color: #ecfdf5 !important;
      padding: 1.1rem 1.2rem 1.15rem 1rem !important;
    }
    [data-testid="stSuccess"] p,
    [data-testid="stSuccess"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSuccess"] [data-testid="stMarkdownContainer"] {
      color: #e8fff4 !important;
      font-size: 1.06rem !important;
      line-height: 1.65 !important;
      margin: 0 !important;
    }
    [data-testid="stSuccess"] strong {
      color: #6ee7b7 !important;
      font-weight: 700 !important;
    }
    [data-testid="stSuccess"] svg,
    [data-testid="stSuccess"] [data-testid="stIcon"] {
      color: #34d399 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ UI ------------------
st.markdown(
    '<h1 style="text-align: center; font-size: 2.75rem; font-weight: 700; margin: 0.25rem 0 0.5rem 0; line-height: 1.2;">'
    "🌿 Plant disease recognition from leaf images</h1>",
    unsafe_allow_html=True,
)

_home = _find_image("home_page")
if _home is not None:
    _lc, _mc, _rc = st.columns([1, 3, 1])
    with _mc:
        st.image(str(_home), use_container_width=True)
else:
    st.caption(
        f"To show a hero image, add `home_page.jpeg` (or `.jpg` / `.png`) in `{SCRIPT_DIR.name}/` next to this app."
    )

st.markdown("### Caring for plants and staying a step ahead of problems")
st.markdown(
    """
Healthy plants need the **right light**, **water**, and **soil** for their species, plus good **air flow**
and room to grow: match each plant to its natural preferences (drainage, brightness, and humidity) instead
of one-size-fits-all watering. **Spacing and gentle pruning** keep canopies open so leaves dry after rain
or misting, which makes life harder for many fungi and some pests. Build a small routine to **check leaves
and stems often** for spots, color shifts, holes, stickiness, webs, or powdery films—problems that stay
hidden until half the plant looks wrong are much harder to reverse. **Sanitation matters** too: take off
grossly infected material when that is safe for the plant, avoid splashing soil onto foliage, and clean
pruners when moving between plants. When the stakes are high—food security, a commercial crop, a rare
specimen, or any symptom that puzzles you—treat the web and books as a starting point and **get local
advice** from an extension agent, agronomist, or plant health clinic; they can tie symptoms to your weather,
pests, and rules in a way a photo alone never will.
    """
)

st.markdown("### What this application does")
st.markdown(
    """
**We turn a single photo of a leaf into a data point for a trained image classifier** so you are not
guessing in a vacuum. You upload a clear picture; we **resize and normalize** it the same way the model
was trained, run the neural network, and return the **label it thinks fits best** from a long list of crop
and condition classes in our dataset. The interface you are using is meant to be **fast and repeatable**:
the same file always follows the same steps on your machine, using the Keras model file you placed beside
this app. **In a typical local setup, inference runs on your own computer** with the model file you add next to
this app—so your image is not sent to a separate service by this code, unless you change the deployment
yourself.

**This remains a decision-support tool, not a licensed diagnosis, treatment plan, or compliance guarantee.**
The model can miss rare diseases, mix up similar patterns, or fail on blurry, backlit, or non-leaf
subjects. It never sees the whole field, the roots, the soil, or the weather that week. **Use the output to
steer your next questions** (what to research, which expert to call, when to send a sample to a lab),
not to replace human judgment, legal rules, or professional scouting on commercial farms. We built it to
**narrow possibilities quickly**; confirming what is actually happening in the plant and in the law is
still on you and your local experts.
    """
)

# ------------------ IMAGE GALLERY ------------------
st.subheader("Examples")
c1, c2, c3 = st.columns(3)

for col, (key, title, _) in zip((c1, c2, c3), _GALLERY):
    path = _find_image(key)
    with col:
        if path:
            st.image(str(path), use_container_width=True, caption=title)
        else:
            st.info(f"Add `{key}.jpg` to show image")

st.divider()

# ------------------ UPLOAD (symmetric columns = centered on page) ------------------
_sec_l, _sec_c, _sec_r = st.columns([1, 1.65, 1])
with _sec_c:
    st.markdown(
        "<h3 style='text-align: center; margin: 0 0 0.35rem; color: #ffffff;'>Check your leaf</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #9ca3af; font-size: 0.92rem; margin: 0 0 1rem; line-height: 1.55;'>"
        "Add one <strong style='color:#e5e7eb;'>leaf</strong> photo, check the preview, then run the model. "
        "Use bright, even light and get close enough that the leaf fills most of the frame.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #d1d5db; font-size: 1.02rem; margin: 0 0 0.5rem;'>"
        "Choose a leaf photo</p>",
        unsafe_allow_html=True,
    )
    with st.container(border=True, key="pd_leaf_upload"):
        uploaded = st.file_uploader(
            "Upload",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
            help="PNG, JPG, JPEG, or WebP. Up to 200MB per file (Streamlit default).",
        )

    if uploaded is not None:
        st.markdown(
            "<p style='text-align: center; color: #9ca3af; font-size: 0.82rem; margin: 1rem 0 0.4rem;'>"
            "Preview</p>",
            unsafe_allow_html=True,
        )
        _pv_l, _pv_c, _pv_r = st.columns([0.2, 1, 0.2])
        with _pv_c:
            st.image(BytesIO(uploaded.getvalue()), use_container_width=True)

    st.markdown("")
    _b_left, _b_mid, _b_right = st.columns([0.35, 1, 0.35])
    with _b_mid:
        run = st.button(
            "🌿 Analyze this leaf",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )
    if run:
        if uploaded is None:
            st.error("Please choose an image file first.")
        else:
            with st.spinner("Running the model on your image…"):
                idx = model_prediction(uploaded)
            raw = class_name[idx]
            st.success(_prediction_narration(raw))