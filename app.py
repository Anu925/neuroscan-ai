import streamlit as st
import numpy as np
import cv2
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;
}

.stApp {
    background: linear-gradient(135deg, #080c14 0%, #0d1421 50%, #080c14 100%);
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Hero Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(56, 189, 248, 0.15);
    margin-bottom: 2rem;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #38bdf8);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    margin: 0;
    letter-spacing: -1px;
}

@keyframes shimmer {
    to { background-position: 200% center; }
}

.hero-subtitle {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Cards */
.info-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

.info-card h4 {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #38bdf8;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 0.5rem 0;
}

.info-card p {
    font-size: 0.9rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.6;
}

/* Result cards */
.result-positive {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(15,23,42,0.9));
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.result-negative {
    background: linear-gradient(135deg, rgba(34,197,94,0.08), rgba(15,23,42,0.9));
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.result-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0.5rem 0;
}

.result-positive .result-title { color: #ef4444; }
.result-negative .result-title { color: #22c55e; }

.result-sub {
    font-size: 0.85rem;
    color: #64748b;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Metric boxes */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}

.metric-box {
    flex: 1;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(56, 189, 248, 0.1);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #38bdf8;
}

.metric-label {
    font-size: 0.72rem;
    color: #475569;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: rgba(15, 23, 42, 0.4);
    margin-bottom: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(8, 12, 20, 0.95);
    border-right: 1px solid rgba(56, 189, 248, 0.08);
}

section[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Space Mono', monospace;
    color: #38bdf8;
    font-size: 1rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 12px;
    padding: 0.5rem;
}

/* Spinner */
.stSpinner > div { border-top-color: #38bdf8 !important; }

/* Divider */
hr { border-color: rgba(56, 189, 248, 0.1); }

/* Legend dot */
.legend-dot {
    display: inline-block;
    width: 12px; height: 12px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_config():
    """Load the trained U-Net model and threshold config."""
    import tensorflow as tf

    model_path  = "brain_tumor_unet.keras"
    config_path = "model_config.json"

    if not os.path.exists(model_path):
        return None, None, None

    # Custom objects needed for loading
    def dice_coefficient(y_true, y_pred, smooth=1.0):
        import tensorflow.keras.backend as K
        y_true_f = K.flatten(tf.cast(y_true, tf.float32))
        y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_loss(y_true, y_pred):
        return 1.0 - dice_coefficient(y_true, y_pred)

    def combined_loss(y_true, y_pred):
        import tensorflow.keras.losses as losses
        bce = losses.binary_crossentropy(y_true, y_pred)
        return bce + dice_loss(y_true, y_pred)

    def iou_score(y_true, y_pred, smooth=1.0):
        import tensorflow.keras.backend as K
        y_true_f = K.flatten(tf.cast(y_true, tf.float32))
        y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)

    custom_objects = {
        "dice_coefficient": dice_coefficient,
        "dice_loss": dice_loss,
        "combined_loss": combined_loss,
        "iou_score": iou_score
    }

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    threshold = 0.3
    img_size  = 128
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        threshold = cfg.get("threshold", 0.3)
        img_size  = cfg.get("img_size", 128)

    return model, threshold, img_size


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_tumor(model, image_input, img_size=128, threshold=0.3):
    """
    Run tumour segmentation on any MRI input.
    Accepts grayscale (H,W), (H,W,1), RGB (H,W,3), or 4-channel (H,W,4).
    Returns: binary_mask (H,W), raw_output (H,W), confidence (float), tumor_detected (bool)
    """
    img = image_input.copy().astype(np.float32)

    # Normalise to [0, 1] if needed
    if img.max() > 1.0:
        img = img / 255.0

    # Ensure (H, W, 4)
    if img.ndim == 2:
        img = np.stack([img] * 4, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.concatenate([img] * 4, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img  = np.stack([gray, gray, gray, img[..., 0]], axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 4:
        pass
    else:
        img = np.stack([img[..., 0]] * 4, axis=-1)

    # Resize each channel
    resized = np.stack([
        cv2.resize(img[..., c], (img_size, img_size))
        for c in range(4)
    ], axis=-1)

    # Per-sample min-max normalise
    mn, mx = resized.min(), resized.max()
    if mx > mn:
        resized = (resized - mn) / (mx - mn)
    else:
        resized = np.zeros_like(resized)

    batch  = resized[np.newaxis, ...]
    output = model.predict(batch, verbose=0)[0]   # (H, W, 3)

    raw_combined  = np.max(output, axis=-1)
    binary_mask   = (raw_combined > threshold).astype(np.uint8)

    tumor_pixels   = int(np.sum(binary_mask))
    tumor_detected = tumor_pixels > 50

    confidence = float(np.mean(raw_combined[binary_mask == 1])) if tumor_detected else 0.0

    # Per-region masks
    region_masks = {
        "Necrotic Core":        (output[..., 0] > threshold).astype(np.uint8),
        "Peritumoral Edema":    (output[..., 1] > threshold).astype(np.uint8),
        "Enhancing Tumour":     (output[..., 2] > threshold).astype(np.uint8),
    }

    return binary_mask, raw_combined, confidence, tumor_detected, region_masks, tumor_pixels


def make_overlay_figure(original_gray, binary_mask, raw_output, region_masks):
    """Generate a 4-panel matplotlib figure for display."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor('#080c14')

    titles  = ["MRI Input", "Segmentation Mask", "Confidence Heatmap", "Region Overlay"]
    colors_bg = '#080c14'

    # Panel 1 — original
    axes[0].imshow(original_gray, cmap='gray')
    axes[0].set_title(titles[0], color='#94a3b8', fontsize=11, pad=10)

    # Panel 2 — binary mask
    axes[1].imshow(binary_mask, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(titles[1], color='#94a3b8', fontsize=11, pad=10)

    # Panel 3 — heatmap
    axes[2].imshow(original_gray, cmap='gray')
    hm = axes[2].imshow(raw_output, cmap='jet', alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title(titles[2], color='#94a3b8', fontsize=11, pad=10)
    plt.colorbar(hm, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4 — colour-coded regions
    axes[3].imshow(original_gray, cmap='gray')
    region_colors = [(1, 0.2, 0.2), (1, 0.8, 0.1), (0.2, 0.8, 1.0)]
    region_names  = list(region_masks.keys())
    patches = []
    for idx, (name, rmask) in enumerate(region_masks.items()):
        if rmask.sum() > 0:
            colored = np.zeros((*rmask.shape, 4))
            c = region_colors[idx]
            colored[rmask == 1] = [*c, 0.7]
            axes[3].imshow(colored)
            patches.append(mpatches.Patch(color=region_colors[idx], label=name))
    if patches:
        axes[3].legend(handles=patches, loc='lower right',
                       fontsize=7, facecolor='#0d1421', edgecolor='#334155',
                       labelcolor='white')
    axes[3].set_title(titles[3], color='#94a3b8', fontsize=11, pad=10)

    for ax in axes:
        ax.axis('off')
        ax.set_facecolor(colors_bg)

    plt.tight_layout(pad=1.5)
    fig.patch.set_facecolor('#080c14')
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 NeuroScan AI")
    st.markdown("---")

    st.markdown("""
    <div class="info-card">
        <h4>About</h4>
        <p>Deep learning brain tumour segmentation using U-Net trained on BraTS 2020 dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>Supported Formats</h4>
        <p>PNG, JPG, JPEG — Upload any MRI brain scan slice. Both grayscale and RGB images are accepted.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>Output Regions</h4>
        <p>
        🔴 <b>Necrotic Core</b> — dead tumour tissue<br><br>
        🟡 <b>Peritumoral Edema</b> — surrounding swelling<br><br>
        🔵 <b>Enhancing Tumour</b> — active tumour
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>Model Info</h4>
        <p>Architecture: U-Net<br>
        Dataset: BraTS 2020<br>
        Input: 128×128×4<br>
        Loss: BCE + Dice</p>
    </div>
    """, unsafe_allow_html=True)


# ── Main page ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <p class="hero-title">🧠 NeuroScan AI</p>
    <p class="hero-subtitle">Brain Tumour Segmentation · U-Net · BraTS 2020</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Loading model..."):
    model, threshold, img_size = load_model_and_config()

if model is None:
    st.error("""
    ⚠️ **Model file not found.**

    Please make sure `brain_tumor_unet.keras` and `model_config.json`
    are in the same folder as `app.py`.
    """)
    st.stop()

st.success(f"✅ Model loaded — detection threshold: **{threshold:.2f}**")

st.markdown("---")

# Upload
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown("### Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose an MRI brain scan image",
        type=["png", "jpg", "jpeg"],
        help="Upload a 2D MRI slice in PNG or JPG format"
    )

with col_info:
    st.markdown("### How It Works")
    st.markdown("""
    1. Upload any MRI brain scan
    2. The U-Net model analyses the image
    3. Tumour regions are segmented and colour-coded
    4. Results show confidence and pixel coverage
    """)

# ── Run inference ─────────────────────────────────────────────────────────────
if uploaded_file is not None:
    st.markdown("---")

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    with st.spinner("Analysing MRI scan..."):
        binary_mask, raw_output, confidence, tumor_detected, region_masks, tumor_pixels = predict_tumor(
            model, img_rgb, img_size=img_size, threshold=threshold
        )

        # Resize outputs to original image size for display
        h, w = img_gray.shape
        binary_mask_disp = cv2.resize(binary_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        raw_output_disp  = cv2.resize(raw_output, (w, h))
        region_masks_disp = {
            k: cv2.resize(v.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            for k, v in region_masks.items()
        }

    # ── Result banner ─────────────────────────────────────────────────────────
    st.markdown("### Detection Result")

    if tumor_detected:
        coverage = (tumor_pixels / (img_size * img_size)) * 100
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-sub">Analysis Complete</div>
            <div class="result-title">⚠️ Tumour Detected</div>
            <div class="result-sub">Segmentation map generated successfully</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{confidence:.1%}</div>
                <div class="metric-label">Confidence</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{tumor_pixels:,}</div>
                <div class="metric-label">Tumour Pixels</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{coverage:.1f}%</div>
                <div class="metric-label">Area Coverage</div>
            </div>""", unsafe_allow_html=True)

        # Region breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Region Breakdown:**")
        rcols = st.columns(3)
        region_colors_hex = ["#ef4444", "#eab308", "#38bdf8"]
        for i, (rname, rmask) in enumerate(region_masks.items()):
            rpixels = int(np.sum(rmask > 0))
            with rcols[i]:
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size:1.1rem;font-weight:700;color:{region_colors_hex[i]}">{rpixels:,} px</div>
                    <div class="metric-label">{rname}</div>
                </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-negative">
            <div class="result-sub">Analysis Complete</div>
            <div class="result-title">✅ No Tumour Detected</div>
            <div class="result-sub">No significant tumour region identified in this scan</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Visualisation panels ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Segmentation Analysis")

    fig = make_overlay_figure(img_gray, binary_mask_disp, raw_output_disp, region_masks_disp)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Download mask ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Download Results")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        mask_img = Image.fromarray((binary_mask_disp * 255).astype(np.uint8))
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Segmentation Mask",
            data=buf.getvalue(),
            file_name="tumor_mask.png",
            mime="image/png"
        )

    with col_dl2:
        buf2 = io.BytesIO()
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        fig2.patch.set_facecolor('#080c14')
        ax2.imshow(img_gray, cmap='gray')
        ax2.imshow(raw_output_disp, cmap='jet', alpha=0.5)
        ax2.axis('off')
        plt.tight_layout()
        fig2.savefig(buf2, format='PNG', facecolor='#080c14', bbox_inches='tight')
        plt.close(fig2)
        st.download_button(
            label="⬇️ Download Heatmap Overlay",
            data=buf2.getvalue(),
            file_name="tumor_heatmap.png",
            mime="image/png"
        )

else:
    # Placeholder when no image uploaded
    st.markdown("""
    <div class="upload-zone">
        <p style="font-size:3rem;margin:0">🧠</p>
        <p style="color:#475569;font-size:1rem;margin:0.5rem 0 0">
            Upload an MRI scan above to begin analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
