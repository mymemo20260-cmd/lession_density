import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import gc
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from ultralytics import YOLO

# ==========================================================
# CONFIGURATION GÉNÉRALE
# ==========================================================

st.set_page_config(
    page_title="Analyse Mammaire Complète",
    page_icon="🩺",
    layout="wide"
)

# ==========================================================
# CONFIGURATION DENSITÉ
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
# CONFIGURATION BI-RADS
# ==========================================================

MODELE_PATH = "detecteur_masscalcif.pt"
IMGSZ = 1024
MARGE_CROP = 0.15

BIRADS_RECO = {
    "BI-RADS 1": "✅ Mammographie normale — aucune lésion détectée.",
    "BI-RADS 2": "🟢 Anomalie bénigne — surveillance standard.",
    "BI-RADS 3": "🟡 Anomalie probablement bénigne — surveillance court terme.",
    "BI-RADS 4": "🔴 Anomalie suspecte — biopsie recommandée.",
}

COULEURS_BIRADS = {
    "BI-RADS 1": "#00FF00",
    "BI-RADS 2": "#00CC00",
    "BI-RADS 3": "#FFA500",
    "BI-RADS 4": "#FF0000",
}

# ==========================================================
# STRUCTURES DE DONNÉES
# ==========================================================

@dataclass
class Lesion:
    type_lesion: str
    birads: str
    details: dict = field(default_factory=dict)
    box: Optional[tuple] = None

@dataclass
class Rapport:
    fichier: str
    nb_masses: int
    nb_amas_calcif: int
    lesions: List[dict]
    birads_final: str
    recommandation: str

@dataclass
class RapportGlobal:
    fichier: str
    densite_acr: str
    densite_confiance: float
    densite_probs: dict
    birads_final: str
    birads_recommandation: str
    nb_masses: int
    nb_calcifications: int
    lesions: List[dict]
    risque_global: str

# ==========================================================
# LOSS ORDINALE (DENSITÉ)
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
# PRÉTRAITEMENT DENSITÉ
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
# CHARGEMENT MODÈLE DENSITÉ
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
# PRÉDICTION DENSITÉ
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
# MASQUE DU SEIN (BI-RADS)
# ==========================================================

def masque_sein(img_gray):
    """Crée un masque du sein pour l'analyse."""
    _, m = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.ones_like(img_gray) * 255
    
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_gray)
    cv2.drawContours(mask, [c], -1, 255, -1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    return cv2.erode(mask, k2, iterations=2)

# ==========================================================
# ANALYSE BI-RADS
# ==========================================================

def features(contour):
    """Extrait les caractéristiques géométriques d'un contour."""
    aire = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim == 0 or aire == 0:
        return {}
    
    circularite = 4 * np.pi * aire / (perim ** 2)
    hull = cv2.convexHull(contour)
    ah = cv2.contourArea(hull)
    convexite = aire / ah if ah > 0 else 0
    deficit_convexite = (ah - aire) / ah if ah > 0 else 0
    
    rapport_axes = 1.0
    if len(contour) >= 5:
        try:
            e = cv2.fitEllipse(contour)
            a, b = max(e[1]), min(e[1])
            rapport_axes = a / b if b > 0 else 1
        except:
            pass
    
    approx = cv2.approxPolyDP(contour, 0.005 * perim, True)
    rugosite = len(contour) / max(len(approx), 1)
    
    return {
        "circularite": circularite,
        "convexite": convexite,
        "deficit_convexite": deficit_convexite,
        "rapport_axes": rapport_axes,
        "rugosite": rugosite
    }

def classer_masse(features):
    """Classe la forme de la masse."""
    if not features:
        return "irreguliere"
    
    if features["circularite"] >= 0.80 and features["rapport_axes"] < 1.4:
        return "ronde_arrondie"
    elif features["circularite"] >= 0.55 and features["rapport_axes"] < 2.5 and features["convexite"] > 0.80:
        return "ovale_elliptique"
    return "irreguliere"

def analyser_masse(crop):
    """Analyse une masse détectée."""
    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cont, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cont:
        return {"forme": "indistinct", "score": 0.3}
    
    c = max(cont, key=cv2.contourArea)
    feat = features(c)
    forme = classer_masse(feat)
    
    score = 0.1 if forme == "ronde_arrondie" else 0.3 if forme == "ovale_elliptique" else 0.5
    
    return {"forme": forme, "score": round(min(score, 1.0), 2)}

def detecter_masses(img_gray, model, conf):
    """Détecte les masses avec YOLO."""
    img_3c = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    res = model.predict(img_3c, conf=conf, iou=0.5, imgsz=IMGSZ, verbose=False)[0]
    masses = []
    
    if res.boxes is not None:
        for box in res.boxes:
            if int(box.cls[0]) != 0:
                continue
            
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            crop = img_gray[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            analyse = analyser_masse(crop)
            
            score = analyse["score"] + (1 - float(box.conf[0])) * 0.1
            if score >= 0.45:
                birads = "BI-RADS 4"
            elif score >= 0.25:
                birads = "BI-RADS 3"
            else:
                birads = "BI-RADS 2"
            
            masses.append(Lesion(
                "Masse",
                birads,
                {
                    "forme": analyse["forme"],
                    "confiance": round(float(box.conf[0]), 2),
                    "score": score
                },
                (x1, y1, x2, y2)
            ))
    
    return masses

def detecter_calcifications(img_gray):
    """Détecte les microcalcifications."""
    mask = masque_sein(img_gray)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(img_gray)
    
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    tophat = cv2.morphologyEx(enh, cv2.MORPH_TOPHAT, se)
    
    zone = tophat[mask > 0]
    seuil = np.percentile(zone, 99) if zone.size else 30
    _, thr = cv2.threshold(tophat, seuil, 255, cv2.THRESH_BINARY)
    thr = cv2.bitwise_and(thr, thr, mask=mask)
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    for c in contours:
        aire = cv2.contourArea(c)
        if 2 <= aire <= 90:
            M = cv2.moments(c)
            if M["m00"] > 0:
                points.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    
    return points

def regrouper_amas(points, img_w):
    """Regroupe les calcifications en amas."""
    if len(points) < 3:
        return []
    
    pts = np.array(points)
    rayon = 0.06 * img_w
    indices = list(range(len(pts)))
    amas = []
    
    while indices:
        base = indices.pop(0)
        groupe = [base]
        i = 0
        while i < len(indices):
            idx = indices[i]
            if any(np.linalg.norm(pts[idx] - pts[g]) < rayon for g in groupe):
                groupe.append(idx)
                indices.pop(i)
                i = 0
            else:
                i += 1
        
        if len(groupe) >= 3:
            amas.append([tuple(pts[g]) for g in groupe])
    
    return amas

def analyser_birads(img_gray, model, conf):
    """Pipeline complet d'analyse BI-RADS."""
    H, W = img_gray.shape
    
    masses = detecter_masses(img_gray, model, conf)
    points = detecter_calcifications(img_gray)
    amas = regrouper_amas(points, W)
    
    lesions = list(masses)
    for amas_pts in amas:
        xs = [p[0] for p in amas_pts]
        ys = [p[1] for p in amas_pts]
        box = (int(min(xs)) - 10, int(min(ys)) - 10, int(max(xs)) + 10, int(max(ys)) + 10)
        
        nb_calcif = len(amas_pts)
        if nb_calcif >= 10:
            birads = "BI-RADS 3"
        elif nb_calcif >= 3:
            birads = "BI-RADS 2"
        else:
            birads = "BI-RADS 2"
        
        lesions.append(Lesion(
            "Calcifications",
            birads,
            {"nombre": nb_calcif, "taille": "micro"},
            box
        ))
    
    if not lesions:
        birads_final = "BI-RADS 1"
    else:
        niveaux = {"BI-RADS 1": 0, "BI-RADS 2": 2, "BI-RADS 3": 3, "BI-RADS 4": 4}
        max_niveau = max(niveaux.get(l.birads, 0) for l in lesions)
        birads_final = f"BI-RADS {max_niveau}" if max_niveau > 0 else "BI-RADS 1"
    
    return Rapport(
        "image",
        len(masses),
        len(amas),
        [asdict(l) for l in lesions],
        birads_final,
        BIRADS_RECO.get(birads_final, "")
    )

def annoter_image_birads(img_gray, rapport):
    """Annote l'image avec les détections BI-RADS."""
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    couleurs = {"BI-RADS 2": (0, 180, 0), "BI-RADS 3": (0, 180, 255), "BI-RADS 4": (0, 0, 220)}
    
    for l in rapport.lesions:
        if not l.get("box"):
            continue
        
        x1, y1, x2, y2 = l["box"]
        c = couleurs.get(l["birads"], (200, 200, 200))
        
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
        label = f"{l['type_lesion']} {l['birads']}"
        cv2.putText(img, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ==========================================================
# ANALYSE COMPLÈTE
# ==========================================================

def analyse_complete(img, density_model, birads_model):
    """Analyse complète avec densité + BI-RADS."""
    results = {"success": True}
    
    try:
        # Conversion en gris pour BI-RADS
        if len(np.array(img).shape) == 3:
            img_gray = np.array(img.convert('L'))
        else:
            img_gray = np.array(img)
        
        # Analyse densité
        img_array_preprocessed = preprocess_pil_image(img, IMG_SIZE_PREPROC)
        results["preprocessed"] = img_array_preprocessed
        
        arr_density = prepare_for_model(img_array_preprocessed, IMG_SIZE_DENSITY)
        label, conf, probs = predict_density(density_model, arr_density)
        
        results.update({
            "density_label": label,
            "density_confidence": conf,
            "density_probs": probs
        })
        
        # Analyse BI-RADS
        birads_rapport = analyser_birads(img_gray, birads_model, 0.15)
        img_annotee = annoter_image_birads(img_gray, birads_rapport)
        
        results.update({
            "birads_rapport": birads_rapport,
            "img_annotee": img_annotee
        })
        
        # Évaluation du risque global
        risque = evaluer_risque_global(label, birads_rapport)
        results["risque_global"] = risque
        
    except Exception as e:
        results = {"success": False, "error": str(e)}
    
    return results

def evaluer_risque_global(density_label, birads_rapport):
    """Évalue le risque global combiné."""
    niveau_risque = 0
    
    # Facteur densité
    if density_label == "ACR_D":
        niveau_risque += 2
    elif density_label == "ACR_C":
        niveau_risque += 1
    
    # Facteur BI-RADS
    if birads_rapport.birads_final == "BI-RADS 4":
        niveau_risque += 3
    elif birads_rapport.birads_final == "BI-RADS 3":
        niveau_risque += 2
    elif birads_rapport.birads_final == "BI-RADS 2":
        niveau_risque += 1
    
    if niveau_risque >= 4:
        return "⚠️ Risque Élevé - Consultation spécialiste recommandée"
    elif niveau_risque >= 2:
        return "🟡 Risque Modéré - Surveillance rapprochée"
    else:
        return "✅ Risque Faible - Surveillance standard"

# ==========================================================
# INTERFACE STREAMLIT
# ==========================================================

def main():
    # CSS personnalisé
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton > button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            font-size: 1.2rem;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.8rem;
        }
        .stButton > button:hover {
            background-color: #FF6B6B;
            color: white;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .birads-1 { color: #00FF00; font-weight: bold; }
        .birads-2 { color: #00CC00; font-weight: bold; }
        .birads-3 { color: #FFA500; font-weight: bold; }
        .birads-4 { color: #FF0000; font-weight: bold; }
        .info-box {
            background-color: #e8f4f8;
            border-left: 5px solid #2196F3;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .risque-eleve { background-color: #FF0000; color: white; padding: 10px; border-radius: 5px; }
        .risque-modere { background-color: #FFA500; color: white; padding: 10px; border-radius: 5px; }
        .risque-faible { background-color: #00CC00; color: white; padding: 10px; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)
    
    # En-tête
    st.title("🩺 Analyse Mammaire Complète")
    st.caption("Classification ACR + Détection BI-RADS")
    st.markdown("---")
    
    # Chargement des modèles
    with st.spinner("Chargement des modèles..."):
        density_model, density_error = load_density_model()
        
        @st.cache_resource
        def load_birads_model():
            try:
                return YOLO(MODELE_PATH)
            except Exception as e:
                return None
        
        birads_model = load_birads_model()
    
    if density_model is None:
        st.error(f"❌ Erreur chargement modèle densité: {density_error}")
        st.stop()
    
    if birads_model is None:
        st.error("❌ Erreur chargement modèle BI-RADS")
        st.stop()
    
    st.success("✅ Modèles chargés avec succès")
    
    # Upload
    uploaded_file = st.file_uploader(
        "📤 Choisissez une mammographie",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="Formats supportés: JPG, JPEG, PNG, BMP, TIFF, WEBP"
    )
    
    if uploaded_file is None:
        st.info("👆 Téléchargez une image pour commencer l'analyse")
        with st.expander("ℹ️ Guide d'utilisation"):
            st.markdown("""
            **Pipeline d'analyse :**
            1. **Classification ACR** — Densité mammaire (B/C/D)
            2. **Détection BI-RADS** — Lésions et calcifications
            3. **Évaluation du risque global**
            """)
        return
    
    # Chargement de l'image
    pil_img = Image.open(uploaded_file)
    img_array = np.array(pil_img)
    
    # Affichage de l'image
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 📷 Image originale")
        if len(img_array.shape) == 3:
            st.image(img_array, use_container_width=True)
        else:
            st.image(img_array, use_container_width=True, channels="GRAY")
    
    with col_right:
        st.markdown("### 🔬 Analyse")
        
        if st.button("🚀 Lancer l'analyse complète", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                results = analyse_complete(pil_img, density_model, birads_model)
            
            if not results["success"]:
                st.error(f"❌ Erreur: {results.get('error', 'Inconnue')}")
                return
            
            # Affichage des résultats
            st.image(results["img_annotee"], use_container_width=True, caption="Détections BI-RADS")
    
    # ==========================================================
    # RAPPORT GLOBAL
    # ==========================================================
    
    st.markdown("---")
    st.markdown("## 📋 Rapport Global d'Analyse")
    
    if "results" in locals() and results["success"]:
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Densité ACR", results["density_label"], 
                     help=f"{DENSITY_DESC.get(results['density_label'], '')}")
        
        with col2:
            st.metric("BI-RADS", results["birads_rapport"].birads_final,
                     help=results["birads_rapport"].recommandation)
        
        with col3:
            nb_lesions = len(results["birads_rapport"].lesions)
            st.metric("Lésions détectées", nb_lesions)
        
        with col4:
            conf = results["density_confidence"] * 100
            st.metric("Confiance densité", f"{conf:.1f}%")
        
        # Risque global
        st.markdown("### 🎯 Évaluation du risque")
        risque = results["risque_global"]
        if "Élevé" in risque:
            st.markdown(f'<div class="risque-eleve">{risque}</div>', unsafe_allow_html=True)
        elif "Modéré" in risque:
            st.markdown(f'<div class="risque-modere">{risque}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risque-faible">{risque}</div>', unsafe_allow_html=True)
        
        # Détails densité
        with st.expander("📊 Détails de la densité", expanded=False):
            st.markdown("**Probabilités par classe:**")
            probs = results["density_probs"]
            for cls in ["ACR_B", "ACR_C", "ACR_D"]:
                prob = probs.get(cls, 0) * 100
                color = DENSITY_COLORS.get(cls, "#999")
                st.progress(prob / 100, text=f"{cls}: {prob:.1f}%")
                st.markdown(f'<span style="color:{color}">■</span> {DENSITY_DESC.get(cls, "")}', 
                           unsafe_allow_html=True)
        
        # Détails BI-RADS
        with st.expander("🔬 Détails des lésions BI-RADS", expanded=False):
            if results["birads_rapport"].lesions:
                data = []
                for i, l in enumerate(results["birads_rapport"].lesions, 1):
                    data.append({
                        "N°": i,
                        "Type": l["type_lesion"],
                        "BI-RADS": l["birads"],
                        "Détails": str(l["details"])[:50] + "..."
                    })
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ Aucune lésion détectée")
        
        # Recommandation
        st.markdown("### 💡 Recommandation clinique")
        st.markdown(f"""
            <div class="info-box">
                <strong>BI-RADS:</strong> {results['birads_rapport'].recommandation}<br>
                <strong>Densité:</strong> {DENSITY_DESC.get(results['density_label'], '')}<br>
                <strong>Risque global:</strong> {risque}
            </div>
        """, unsafe_allow_html=True)
        
        # Export
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Exporter Rapport (JSON)", use_container_width=True):
                rapport_global = {
                    "fichier": uploaded_file.name,
                    "densite": {
                        "acr": results["density_label"],
                        "confiance": results["density_confidence"],
                        "probabilites": results["density_probs"]
                    },
                    "birads": {
                        "classification": results["birads_rapport"].birads_final,
                        "recommandation": results["birads_rapport"].recommandation,
                        "nb_masses": results["birads_rapport"].nb_masses,
                        "nb_calcifications": results["birads_rapport"].nb_amas_calcif,
                        "lesions": results["birads_rapport"].lesions
                    },
                    "risque_global": results["risque_global"]
                }
                json_str = json.dumps(rapport_global, indent=2, default=str)
                st.download_button(
                    "📥 Télécharger",
                    data=json_str,
                    file_name=f"rapport_{uploaded_file.name.split('.')[0]}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Exporter Tableau (CSV)", use_container_width=True):
                rows = []
                for l in results["birads_rapport"].lesions:
                    rows.append({
                        "Fichier": uploaded_file.name,
                        "Densité_ACR": results["density_label"],
                        "BI-RADS": results["birads_rapport"].birads_final,
                        "Type_lésion": l["type_lesion"],
                        "BI-RADS_lésion": l["birads"],
                        "Détails": str(l["details"])
                    })
                csv = pd.DataFrame(rows).to_csv(index=False)
                st.download_button(
                    "📥 Télécharger",
                    data=csv,
                    file_name=f"rapport_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv"
                )
    
    # Nettoyage mémoire
    gc.collect()

if __name__ == "__main__":
    main()
