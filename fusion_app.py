import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import gc

# ==========================================================
# CONFIGURATION
# ==========================================================

IMG_SIZE_DENSITY = 224
IMG_SIZE_PREPROC = 512

CLASS_NAMES_DENSITY = {
    0: "ACR_B",
    1: "ACR_C",
    2: "ACR_D"
}

DENSITY_COLORS = {
    "ACR_B": "#4CAF50",
    "ACR_C": "#FF9800",
    "ACR_D": "#F44336"
}

DENSITY_DESC = {
    "ACR_B": "Densité moyenne faible (25–50% glandulaire)",
    "ACR_C": "Densité moyenne élevée (50–75% glandulaire)",
    "ACR_D": "Densité extrême (> 75% glandulaire)"
}

# ==========================================================
# LOSS ORDINALE
# ==========================================================

@tf.keras.utils.register_keras_serializable(package="Custom")
def ordinal_loss(y_true, y_pred):
    weights = tf.constant([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
    ], dtype=tf.float32)
    ce = tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, label_smoothing=0.1
    )
    true_class = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
    pred_class = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
    indices    = tf.stack([true_class, pred_class], axis=1)
    penalty    = tf.gather_nd(weights, indices)
    return ce + 0.5 * tf.cast(penalty, tf.float32)

# ==========================================================
# PRÉTRAITEMENT DES MAMMOGRAPHIES
# ==========================================================

def preprocess_mammogram_image(img_array: np.ndarray, img_size: int = IMG_SIZE_PREPROC) -> np.ndarray:
    """Pipeline de prétraitement adapté aux mammographies."""
    
    if len(img_array.shape) == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array.copy()
    
    # Uniformisation fond blanc/noir
    border_pixels = np.concatenate([
        img[0, :], img[-1, :],
        img[:, 0], img[:, -1]
    ])
    border_mean = np.mean(border_pixels)
    if border_mean > 127:
        img = cv2.bitwise_not(img)
    
    # Détection du sein
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        img = img[y:y+h, x:x+w]
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Sharpen léger
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    
    # Normalisation
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Resize en conservant les proportions
    h, w = img.shape
    scale = min(img_size / w, img_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Canvas noir
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    x_offset = (img_size - new_w) // 2
    y_offset = (img_size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img
    
    return canvas

def preprocess_pil_image(pil_img: Image.Image, img_size: int = IMG_SIZE_PREPROC) -> np.ndarray:
    """Convertit une image PIL en tableau numpy prétraité."""
    img_array = np.array(pil_img)
    return preprocess_mammogram_image(img_array, img_size)

# ==========================================================
# CHARGEMENT DU MODÈLE
# ==========================================================

@st.cache_resource
def load_density_model():
    try:
        model = tf.keras.models.load_model(
            "breast_density_cnn_final.keras",
            compile=False
        )
        return model, None
    except Exception as e:
        return None, str(e)

# ==========================================================
# PRÉDICTION
# ==========================================================

def prepare_for_model(img_array: np.ndarray, target_size: int) -> np.ndarray:
    """Prépare l'image prétraitée pour le modèle CNN."""
    if len(img_array.shape) == 2:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_array
    
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    arr = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_density(model, img_array):
    probs = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASS_NAMES_DENSITY[idx]
    all_probs = {CLASS_NAMES_DENSITY[i]: float(probs[i]) for i in range(3)}
    return label, float(probs[idx]), all_probs

# ==========================================================
# ANALYSE
# ==========================================================

def analyze_density(img, model):
    results = {"success": True}
    try:
        img_array_preprocessed = preprocess_pil_image(img, IMG_SIZE_PREPROC)
        results["preprocessed"] = img_array_preprocessed
        
        arr_density = prepare_for_model(img_array_preprocessed, IMG_SIZE_DENSITY)
        label, conf, probs = predict_density(model, arr_density)
        
        results.update({
            "density_label": label,
            "density_confidence": conf,
            "density_probs": probs
        })
    except Exception as e:
        results = {"success": False, "error": str(e)}
    return results

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Analyse Densité Mammaire",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Analyse Densité Mammaire")
st.caption("Classification ACR (B / C / D) des mammographies")
st.markdown("---")

# ==========================================================
# CHARGEMENT
# ==========================================================

with st.spinner("Chargement du modèle de densité…"):
    density_model, density_error = load_density_model()

if density_model:
    st.success("✅ Modèle de densité chargé avec succès")
else:
    st.error(f"❌ Erreur de chargement : {density_error}")
    st.stop()

# ==========================================================
# UPLOAD
# ==========================================================

uploaded_files = st.file_uploader(
    "📤 Uploader une ou plusieurs mammographies",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👆 Uploadez une ou plusieurs images pour commencer.")
    with st.expander("ℹ️ Guide d'utilisation"):
        st.markdown("""
        **Pipeline d'analyse :**
        1. **Prétraitement** — Uniformisation fond, recadrage du sein, CLAHE, sharpening
        2. **Classification ACR** — B / C / D (DensityCNN-SE avec loss ordinale)
        
        **Classes ACR :**
        - **ACR_B** : Densité moyenne faible (25–50% glandulaire)
        - **ACR_C** : Densité moyenne élevée (50–75% glandulaire)
        - **ACR_D** : Densité extrême (> 75% glandulaire)
        """)
    st.stop()

st.success(f"{len(uploaded_files)} image(s) chargée(s)")

# ==========================================================
# ANALYSE
# ==========================================================

if "results" not in st.session_state:
    st.session_state.results = {}

if st.button("🔍 Analyser la densité", type="primary", use_container_width=True):
    bar = st.progress(0)
    info = st.empty()
    
    for i, f in enumerate(uploaded_files):
        info.text(f"Analyse de {f.name}… ({i+1}/{len(uploaded_files)})")
        img = Image.open(f)
        st.session_state.results[f.name] = analyze_density(img, density_model)
        bar.progress((i + 1) / len(uploaded_files))
    
    bar.empty()
    info.empty()
    st.success("✅ Analyse terminée !")

# ==========================================================
# EXPORT CSV
# ==========================================================

if st.button("📊 Exporter CSV", use_container_width=True) and st.session_state.results:
    rows = []
    for fname, res in st.session_state.results.items():
        row = {"Fichier": fname}
        if res.get("success"):
            row["Statut"] = "OK"
            row["Densité ACR"] = res.get("density_label", "—")
            row["Confiance_%"] = f"{res.get('density_confidence', 0)*100:.1f}"
            # Ajouter les probabilités détaillées
            probs = res.get("density_probs", {})
            for cls in ["ACR_B", "ACR_C", "ACR_D"]:
                row[f"Prob_{cls}"] = f"{probs.get(cls, 0)*100:.1f}%"
        else:
            row["Statut"] = "Erreur"
            row["Erreur"] = res.get("error", "")
        rows.append(row)
    
    csv = pd.DataFrame(rows).to_csv(index=False)
    st.download_button(
        "📥 Télécharger le CSV",
        data=csv,
        file_name="analyse_densite_mammaire.csv",
        mime="text/csv",
        use_container_width=True
    )

# ==========================================================
# AFFICHAGE DES RÉSULTATS
# ==========================================================

st.markdown("---")
view = st.radio(
    "Mode d'affichage",
    ["Grille", "Détail", "Tableau"],
    horizontal=True
)

# ---------- GRILLE ----------

if view == "Grille":
    for i in range(0, len(uploaded_files), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(uploaded_files):
                break
            f = uploaded_files[idx]
            with col:
                st.markdown(f"**{f.name}**")
                
                # Afficher l'image prétraitée si disponible
                res = st.session_state.results.get(f.name)
                if res and res.get("success") and res.get("preprocessed") is not None:
                    st.image(res["preprocessed"], use_container_width=True, clamp=True)
                else:
                    st.image(Image.open(f), use_container_width=True)
                
                if res and res.get("success"):
                    label = res.get("density_label", "—")
                    conf = res.get("density_confidence", 0) * 100
                    color = DENSITY_COLORS.get(label, "#999")
                    desc = DENSITY_DESC.get(label, "")
                    
                    st.markdown(
                        f'<div style="background:{color};padding:10px;border-radius:6px;'
                        f'color:#fff;text-align:center;margin:8px 0">'
                        f'<h3 style="margin:0">{label}</h3>'
                        f'<small>{desc}</small><br>'
                        f'<small>Confiance : {conf:.1f}%</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Barres de progression pour les probabilités
                    probs = res.get("density_probs", {})
                    for cls in ["ACR_B", "ACR_C", "ACR_D"]:
                        prob = probs.get(cls, 0) * 100
                        st.progress(prob / 100, text=f"{cls}: {prob:.1f}%")
                else:
                    st.info("Non analysé — cliquez sur Analyser")

# ---------- DÉTAIL ----------

elif view == "Détail":
    for f in uploaded_files:
        with st.expander(f"📷 {f.name}"):
            res = st.session_state.results.get(f.name)
            
            if res and res.get("success") and res.get("preprocessed") is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("📷 **Original**")
                    st.image(Image.open(f), use_container_width=True)
                with col2:
                    st.caption("🔧 **Prétraitée** (CLAHE + Sharpening)")
                    st.image(res["preprocessed"], use_container_width=True, clamp=True)
            
            if not res:
                st.info("Non analysé.")
            elif not res["success"]:
                st.error(res.get("error", "Erreur inconnue"))
            else:
                label = res.get("density_label", "—")
                conf = res.get("density_confidence", 0) * 100
                color = DENSITY_COLORS.get(label, "#999")
                desc = DENSITY_DESC.get(label, "")
                
                st.markdown(
                    f'<div style="background:{color};padding:20px;border-radius:10px;'
                    f'color:#fff;text-align:center;margin:10px 0">'
                    f'<h2 style="margin:0">{label}</h2>'
                    f'<p style="margin:5px 0">{desc}</p>'
                    f'<p style="margin:5px 0">Confiance : {conf:.1f}%</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                st.subheader("📊 Probabilités détaillées")
                probs = res.get("density_probs", {})
                for cls in ["ACR_B", "ACR_C", "ACR_D"]:
                    prob = probs.get(cls, 0) * 100
                    st.progress(prob / 100, text=f"{cls}: {prob:.1f}%")

# ---------- TABLEAU ----------

else:  # Tableau
    rows = []
    for f in uploaded_files:
        res = st.session_state.results.get(f.name, {})
        row = {"Fichier": f.name}
        if res.get("success"):
            row["Densité ACR"] = res.get("density_label", "—")
            row["Confiance"] = f"{res.get('density_confidence', 0)*100:.1f}%"
            probs = res.get("density_probs", {})
            row["Prob B"] = f"{probs.get('ACR_B', 0)*100:.1f}%"
            row["Prob C"] = f"{probs.get('ACR_C', 0)*100:.1f}%"
            row["Prob D"] = f"{probs.get('ACR_D', 0)*100:.1f}%"
            row["Statut"] = "✅"
        elif res:
            row["Statut"] = "❌"
        else:
            row["Statut"] = "⏳"
        rows.append(row)
    
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    
    # Télécharger le CSV depuis le tableau
    if st.button("📥 Exporter ce tableau en CSV"):
        csv = pd.DataFrame(rows).to_csv(index=False)
        st.download_button(
            "Télécharger",
            data=csv,
            file_name="analyse_densite_tableau.csv",
            mime="text/csv"
        )

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.header("📌 Statistiques")
    
    analyzed = [
        f.name for f in uploaded_files
        if f.name in st.session_state.results
        and st.session_state.results[f.name].get("success")
    ]
    
    st.metric("Images uploadées", len(uploaded_files))
    st.metric("Images analysées", len(analyzed))
    
    if analyzed:
        st.markdown("---")
        st.markdown("**Distribution ACR**")
        for cls in ["ACR_B", "ACR_C", "ACR_D"]:
            count = sum(1 for n in analyzed 
                       if st.session_state.results[n].get("density_label") == cls)
            if count:
                color = DENSITY_COLORS[cls]
                st.markdown(
                    f'<span style="color:{color}">■</span> {cls} : **{count}** ({count/len(analyzed)*100:.0f}%)',
                    unsafe_allow_html=True
                )
    
    st.markdown("---")
    st.markdown("**Modèle**")
    st.markdown("✅ DensityCNN-SE (loss ordinale)")
    st.caption(f"TensorFlow {tf.__version__}  ·  Keras {tf.keras.__version__}")

gc.collect()
