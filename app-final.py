"""
Breast Cancer Histopathology Classification
Interface Streamlit — Deep Learning Demo
"""

import streamlit as st
import numpy as np
from PIL import Image
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS custom ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e6edf3;
}

/* Title */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    color: #e6edf3;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Cards */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 3px solid #f85149;
}
.card-accent-green {
    border-left: 3px solid #3fb950;
}

/* Result badge */
.badge-malin {
    display: inline-block;
    background: rgba(248, 81, 73, 0.15);
    border: 1px solid #f85149;
    color: #f85149;
    border-radius: 20px;
    padding: 0.4rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.badge-benin {
    display: inline-block;
    background: rgba(63, 185, 80, 0.15);
    border: 1px solid #3fb950;
    color: #3fb950;
    border-radius: 20px;
    padding: 0.4rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}

/* Progress bar custom */
.prob-bar-container {
    background: #21262d;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    margin: 0.3rem 0;
}
.prob-bar-malin {
    background: linear-gradient(90deg, #f85149, #ff6b6b);
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}
.prob-bar-benin {
    background: linear-gradient(90deg, #3fb950, #56d364);
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}

/* Model tag */
.model-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #58a6ff;
    background: rgba(88, 166, 255, 0.1);
    border: 1px solid rgba(88, 166, 255, 0.25);
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    display: inline-block;
    margin-bottom: 0.5rem;
}

/* Metrics */
.metric-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #e6edf3;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Divider */
hr {
    border-color: #21262d !important;
    margin: 1.5rem 0 !important;
}

/* Streamlit overrides */
.stSelectbox label, .stFileUploader label {
    color: #8b949e !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
div[data-baseweb="select"] > div {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}
.stButton > button {
    background: #238636 !important;
    border: 1px solid #2ea043 !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #2ea043 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #21262d !important;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    color: #e6edf3 !important;
    border-bottom: 2px solid #f85149 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model config ───────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "VGG16 — Fine Tuning": {
        "path": "models/VGG_model_2.keras",
        "preprocess": "vgg",
        "desc": "VGG16 avec dé-gel des blocs 4 & 5 — meilleure généralisation",
        "badge": "⭐ Recommandé",
    },
    "ResNet50": {
        "path": "models/ResNet50_model.keras",
        "preprocess": "resnet",
        "desc": "ResNet50 Feature Extraction — connexions résiduelles",
        "badge": "🚀 Rapide",
    },
    "VGG16 — Feature Extraction": {
        "path": "models/VGG_model.keras",
        "preprocess": "vgg",
        "desc": "VGG16 backbone gelé — tête Dense entraînée",
        "badge": "",
    },
    "CNN from Scratch": {
        "path": "models/first_model.keras",
        "preprocess": "rescale",
        "desc": "CNN 3 blocs Conv2D — baseline entraîné from scratch",
        "badge": "📐 Baseline",
    },
}

from keras.layers import Dense

# ===== PATCH KERAS =====
original_from_config = Dense.from_config

@classmethod
def patched_from_config(cls, config):
    config.pop("quantization_config", None)
    return original_from_config(config)

Dense.from_config = patched_from_config
@st.cache_resource(show_spinner="Chargement du modèle…")
def load_keras_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(
        path,
        compile=False,
        safe_mode=False
    )

def preprocess_image(img_array, mode):
    if mode == "vgg":
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input(img_array.astype(np.float32))
    elif mode == "resnet":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input(img_array.astype(np.float32))
    else:
        return img_array.astype(np.float32) / 255.0

def run_prediction(image_pil, model_name):
    cfg = MODEL_CONFIG[model_name]
    if not os.path.exists(cfg["path"]):
        return None, f"Modèle introuvable : `{cfg['path']}`"
    model = load_keras_model(cfg["path"])
    img = image_pil.resize((96, 96)).convert("RGB")
    arr = np.array(img)
    arr = preprocess_image(arr, cfg["preprocess"])
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    return prob, None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem 0;">
        <span style="font-family:'DM Serif Display',serif; font-size:1.3rem; color:#e6edf3;">🔬 BC Classifier</span>
        <br/>
        <span style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.1em;">Deep Learning · PCam Dataset</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<p style="font-family:\'DM Mono\',monospace; font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;">Modèle</p>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Modèle",
        options=list(MODEL_CONFIG.keys()),
        index=0,
        label_visibility="collapsed",
    )

    cfg = MODEL_CONFIG[model_choice]
    badge_html = f'<span class="model-tag">{cfg["badge"]}</span><br/>' if cfg["badge"] else ""
    st.markdown(f"""
    <div class="card" style="margin-top:0.5rem;">
        {badge_html}
        <p style="font-size:0.82rem; color:#8b949e; margin:0;">{cfg["desc"]}</p>
        <p style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#30363d; margin-top:0.5rem; word-break:break-all;">{cfg["path"]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#484f58; text-align:center; padding-top:0.5rem;">
        Images : 96×96 px · RGB<br/>
        Tâche : Classification binaire<br/>
        Dataset : PatchCamelyon (PCam)
    </div>
    """, unsafe_allow_html=True)

# ── Main content ───────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">Breast Cancer<br/>Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Histopathologie · Deep Learning · Tissue Analysis</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔬 Analyse", "📊 Modèles", "📖 Méthode"])

# ── Tab 1: Prediction ──────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<p style="font-family:\'DM Mono\',monospace; font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Image histopathologique</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded:
            image_pil = Image.open(uploaded).convert("RGB")
            st.image(image_pil, caption=f"{uploaded.name} · {image_pil.size[0]}×{image_pil.size[1]} px", use_container_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        analyze_btn = st.button("🔬  Lancer l'analyse", disabled=(uploaded is None))

    with col_right:
        st.markdown('<p style="font-family:\'DM Mono\',monospace; font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">Résultat</p>', unsafe_allow_html=True)

        if analyze_btn and uploaded:
            with st.spinner("Analyse en cours…"):
                prob, err = run_prediction(image_pil, model_choice)

            if err:
                st.error(err)
            else:
                is_malin = prob > 0.5
                conf = prob if is_malin else 1 - prob
                label = "Malin" if is_malin else "Bénin"
                badge_class = "badge-malin" if is_malin else "badge-benin"

                pct_malin = round(prob * 100, 1)
                pct_benin = round((1 - prob) * 100, 1)

                st.markdown(f"""
                <div class="card {'card-accent' if is_malin else 'card-accent-green'}">
                    <p style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.8rem;">Diagnostic</p>
                    <span class="{badge_class}">{label}</span>
                    <p style="font-family:'DM Serif Display',serif; font-size:2.4rem; color:#e6edf3; margin:0.8rem 0 0.2rem 0;">{conf*100:.1f}<span style="font-size:1rem; color:#8b949e;">%</span></p>
                    <p style="font-family:'DM Mono',monospace; font-size:0.68rem; color:#8b949e;">confiance</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<p style="font-family:\'DM Mono\',monospace; font-size:0.68rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-top:0.5rem;">Probabilités</p>', unsafe_allow_html=True)

                st.markdown(f"🔴 **Malin** — `{pct_malin}%`")
                st.progress(prob)

                st.markdown(f"🟢 **Bénin** — `{pct_benin}%`")
                st.progress(1 - prob)

                st.markdown(f"""
                <div class="card" style="margin-top:0.8rem; background:#0d1117;">
                    <p style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#484f58; margin:0;">
                        Modèle : {model_choice}<br/>
                        Seuil de décision : 0.5<br/>
                        Score brut (logit) : {prob:.6f}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        elif not uploaded:
            st.markdown("""
            <div class="card" style="text-align:center; padding:3rem 1.5rem;">
                <p style="font-size:2.5rem; margin-bottom:0.5rem;">🧬</p>
                <p style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#484f58; text-transform:uppercase; letter-spacing:0.1em;">
                    Uploadez une image<br/>pour démarrer l'analyse
                </p>
            </div>
            """, unsafe_allow_html=True)

# ── Tab 2: Models comparison ───────────────────────────────────────────────────
with tab2:
    st.markdown('<p style="font-family:\'DM Mono\',monospace; font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:1.5rem;">Architectures entraînées</p>', unsafe_allow_html=True)

    models_data = [
        ("CNN from Scratch",           "3× Conv2D + MaxPool",           "RMSProp",  "1/255",            "Baseline"),
        ("VGG16 — Feature Extraction", "VGG16 (frozen) + Flatten + Dense", "Adam 5e-5", "VGG preprocess", "Transfer Learning"),
        ("VGG16 — Fine Tuning",        "VGG16 (block4-5 unfrozen) + GAP", "Adam 1e-5", "VGG preprocess",  "Fine Tuning"),
        ("ResNet50",                   "ResNet50 (frozen) + GAP",       "Adam 5e-5", "ResNet preprocess","Transfer Learning"),
    ]

    for name, arch, opt, prep, strategy in models_data:
        cfg_m = MODEL_CONFIG[name]
        available = "✅" if os.path.exists(cfg_m["path"]) else "❌ manquant"
        badge = f" · {cfg_m['badge']}" if cfg_m["badge"] else ""

        with st.container(border=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**{name}**{badge}")
                st.caption(arch)
            with col2:
                st.markdown(available)

            c1, c2, c3 = st.columns(3)
            c1.metric("Stratégie", strategy)
            c2.metric("Optimiseur", opt)
            c3.metric("Preprocessing", prep)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'DM Mono\',monospace; font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;">Performances — remplir avec tes résultats</p>', unsafe_allow_html=True)

    perf_data = {
        "Modèle": ["CNN from Scratch", "VGG16 FE", "VGG16 Fine Tuning", "ResNet50"],
        "Accuracy": ["0.843", "0.901", "0.964", "0.756"],
        "F1-Score": ["0.884", "0.924", "0.973", "0.828"],
        "AUC-ROC": ["0.913", "0.962", "0.991", "0.816"],
    }
    st.dataframe(perf_data, use_container_width=True, hide_index=True)

# ── Tab 3: Method ──────────────────────────────────────────────────────────────
with tab3:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("""
        <div class="card">
            <p style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;">Dataset</p>
            <p style="font-size:0.9rem; color:#e6edf3; margin-bottom:0.5rem;"><strong>PatchCamelyon (PCam)</strong></p>
            <p style="font-size:0.82rem; color:#8b949e;">Patches 96×96 px extraits de scans histologiques de ganglions lymphatiques. Tâche : détecter la présence de tissu tumoral dans la région centrale du patch.</p>
            <hr/>
            <div style="display:flex; gap:1.5rem;">
                <div class="metric-box" style="flex:1;"><div class="metric-value">96²</div><div class="metric-label">px / image</div></div>
                <div class="metric-box" style="flex:1;"><div class="metric-value">2</div><div class="metric-label">classes</div></div>
                <div class="metric-box" style="flex:1;"><div class="metric-value">RGB</div><div class="metric-label">canaux</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="margin-top:0.8rem;">
            <p style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;">Augmentation (train)</p>
            <ul style="font-size:0.82rem; color:#8b949e; margin:0; padding-left:1.2rem;">
                <li>Rotation : ±20°</li>
                <li>Flip horizontal & vertical</li>
                <li>Zoom : ±10%</li>
                <li>Rescale : 1/255 (CNN) ou preprocess_input (TL)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card">
            <p style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;">Pipeline d'entraînement</p>
            <div style="font-size:0.82rem; color:#8b949e; line-height:1.9;">
                <span style="color:#58a6ff; font-family:'DM Mono',monospace;">01</span> &nbsp;Chargement depuis Google Drive<br/>
                <span style="color:#58a6ff; font-family:'DM Mono',monospace;">02</span> &nbsp;Générateurs avec augmentation<br/>
                <span style="color:#58a6ff; font-family:'DM Mono',monospace;">03</span> &nbsp;Entraînement (50 epochs max)<br/>
                <span style="color:#58a6ff; font-family:'DM Mono',monospace;">04</span> &nbsp;EarlyStopping (patience=5)<br/>
                <span style="color:#58a6ff; font-family:'DM Mono',monospace;">05</span> &nbsp;ModelCheckpoint (best val_acc)<br/>
                <span style="color:#58a6ff; font-family:'DM Mono',monospace;">06</span> &nbsp;Évaluation : Accuracy / F1 / AUC
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card" style="margin-top:0.8rem;">
            <p style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.8rem;">Structure du projet</p>
            <pre style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#8b949e; background:#0d1117; padding:0.8rem; border-radius:6px; overflow-x:auto;">projet/
├── app.py
├── requirements.txt
└── models/
    ├── first_model.keras
    ├── VGG_model.keras
    ├── VGG_model_2.keras
    └── ResNet50_model.keras</pre>
        </div>
        """, unsafe_allow_html=True)
