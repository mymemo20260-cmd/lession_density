#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Mammographique Complète
================================
Fusion des analyses : Traitement d'image · Densité ACR · Lésions BI-RADS
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import gc
from dataclasses import dataclass
from typing import Optional

# ==========================================================
# CONFIGURATION
# ==========================================================

IMG_SIZE_LR      = 256
IMG_SIZE_DENSITY = 224
IMG_SIZE_PREPROC = 512

CLASS_NAMES_LR = {
    0: "Gauche",
    1: "Droite"
}

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
# BI-RADS CONFIGURATION
# ==========================================================

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
        img[0, :],
        img[-1, :],
        img[:, 0],
        img[:, -1]
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
    
    # Resize
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
# CHARGEMENT DES MODÈLES
# ==========================================================

@st.cache_resource
def load_laterality_model():
    try:
        model = tf.keras.models.load_model(
            "breast_laterality_final.keras",
            compile=False
        )
        return model, None
    except Exception as e:
        return None, str(e)

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
# PRÉTRAITEMENT POUR MODÈLES
# ==========================================================

def prepare_for_model(img_array: np.ndarray, target_size: int) -> np.ndarray:
    """Prépare l'image prétraitée pour les modèles CNN."""
    if len(img_array.shape) == 2:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_array
    
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    arr = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ==========================================================
# PRÉDICTIONS
# ==========================================================

def predict_laterality(model, img_array, threshold=0.5):
    prob  = float(model.predict(img_array, verbose=0)[0][0])
    label = CLASS_NAMES_LR[1] if prob >= threshold else CLASS_NAMES_LR[0]
    conf  = prob if prob >= threshold else 1.0 - prob
    return label, conf, prob

def predict_density(model, img_array):
    probs = model.predict(img_array, verbose=0)[0]
    idx   = int(np.argmax(probs))
    label = CLASS_NAMES_DENSITY[idx]
    all_probs = {CLASS_NAMES_DENSITY[i]: float(probs[i]) for i in range(3)}
    return label, float(probs[idx]), all_probs

# ==========================================================
# DÉTECTION DES LÉSIONS (BI-RADS)
# ==========================================================

@dataclass
class MasseResult:
    detectee:  bool
    forme:     Optional[str]
    contour:   Optional[str]
    densite:   Optional[str]
    nb_masses: int
    score_suspicion: float
    confiance: str

@dataclass
class CalcificationResult:
    detectee:    bool
    forme:       Optional[str]
    repartition: Optional[str]
    benigne:     bool
    nb_calcif:   int
    confiance:   str

@dataclass
class LesionReport:
    fichier:       str
    masse:         MasseResult
    calcification: CalcificationResult
    conclusion:    str
    suspicion:     str
    birads:        str
    recommandation: str

def calculer_features_contour(contour) -> dict:
    aire      = cv2.contourArea(contour)
    perimetre = cv2.arcLength(contour, True)
    
    if perimetre == 0 or aire == 0:
        return {}
    
    circularite = 4 * np.pi * aire / (perimetre ** 2)
    hull = cv2.convexHull(contour)
    aire_hull = cv2.contourArea(hull)
    convexite = aire / aire_hull if aire_hull > 0 else 0
    deficit_convexite = (aire_hull - aire) / aire_hull if aire_hull > 0 else 0
    
    rapport_axes = 1.0
    excentricite = 0.0
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            a, b = max(ellipse[1]), min(ellipse[1])
            rapport_axes = a / b if b > 0 else 1
            excentricite = np.sqrt(1 - (b / a) ** 2) if a > 0 else 0
        except:
            pass
    
    epsilon = 0.005 * perimetre
    approx = cv2.approxPolyDP(contour, epsilon, True)
    rugosite = len(contour) / max(len(approx), 1)
    compacite = perimetre ** 2 / (4 * np.pi * aire)
    
    x, y, w, h = cv2.boundingRect(contour)
    solidite = aire / (w * h) if (w * h) > 0 else 0
    perim_hull = cv2.arcLength(hull, True)
    ratio_perim_hull = perim_hull / perimetre if perimetre > 0 else 1
    
    return {
        "aire": aire, "perimetre": perimetre, "circularite": circularite,
        "convexite": convexite, "deficit_convexite": deficit_convexite,
        "rapport_axes": rapport_axes, "excentricite": excentricite,
        "rugosite": rugosite, "compacite": compacite, "solidite": solidite,
        "ratio_perim_hull": ratio_perim_hull,
    }

def classer_forme(f: dict) -> str:
    if not f:
        return "irreguliere"
    if f["circularite"] >= 0.80 and f["rapport_axes"] < 1.4:
        return "ronde_arrondie"
    elif f["circularite"] >= 0.55 and f["rapport_axes"] < 2.5 and f["convexite"] > 0.80:
        return "ovale_elliptique"
    else:
        return "irreguliere"

def classer_contour_masse(img_gray: np.ndarray, contour, f: dict) -> str:
    if not f:
        return "indistinct"
    
    mask_bord = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask_bord, [contour], -1, 255, 8)
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_bord = gradient[mask_bord > 0]
    grad_moy = np.mean(grad_bord) if len(grad_bord) > 0 else 0
    
    mask_int = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask_int, [contour], -1, 255, -1)
    grad_int = gradient[mask_int > 0]
    grad_int_moy = np.mean(grad_int) if len(grad_int) > 0 else 1
    ratio_grad = grad_moy / max(grad_int_moy, 1)
    
    rugosite = f.get("rugosite", 1)
    deficit_convexite = f.get("deficit_convexite", 0)
    convexite = f.get("convexite", 1)
    
    if rugosite > 25 and deficit_convexite > 0.15:
        return "spicule"
    elif 10 < rugosite <= 25 and convexite < 0.85:
        return "microlobule"
    elif ratio_grad > 1.8 and convexite > 0.90:
        return "circumscrit"
    else:
        return "indistinct"

def classer_densite_masse(img_gray: np.ndarray, contour) -> tuple:
    mask_int = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask_int, [contour], -1, 255, -1)
    
    mask_ext = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask_ext, [contour], -1, 255, 30)
    mask_env = cv2.bitwise_xor(mask_ext, mask_int)
    
    pixels_masse = img_gray[mask_int > 0]
    pixels_env = img_gray[mask_env > 0]
    
    if len(pixels_masse) == 0 or len(pixels_env) == 0:
        return "egale", 1.0
    
    intensite_masse = np.mean(pixels_masse)
    intensite_env = np.mean(pixels_env)
    ratio = intensite_masse / max(intensite_env, 1)
    
    if ratio > 1.30:
        return "haute", ratio
    elif ratio > 1.05:
        return "egale", ratio
    elif ratio > 0.80:
        return "faible", ratio
    else:
        return "graisseuse", ratio

def calculer_score_suspicion(forme: str, contour: str, densite: str) -> float:
    score = 0.0
    if forme == "irreguliere":
        score += 0.40
    elif forme == "ovale_elliptique":
        score += 0.10
    
    if contour == "spicule":
        score += 0.40
    elif contour == "microlobule":
        score += 0.20
    elif contour == "indistinct":
        score += 0.15
    
    if densite == "haute":
        score += 0.20
    elif densite == "egale":
        score += 0.10
    elif densite == "faible":
        score += 0.05
    
    return min(score, 1.0)

def detecter_masses(img_gray: np.ndarray) -> MasseResult:
    h, w = img_gray.shape
    aire_min = int((min(h, w) * 0.03) ** 2)
    aire_max = int((min(h, w) * 0.45) ** 2)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    
    sigma1, sigma2 = 2.0, 20.0
    g1 = cv2.GaussianBlur(enhanced, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(enhanced, (0, 0), sigma2)
    dog = cv2.subtract(g1, g2)
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    block = max(w // 8, 51) | 1
    thresh_adapt = cv2.adaptiveThreshold(dog_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, block, -5)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(thresh_adapt, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidats = []
    for c in contours:
        aire = cv2.contourArea(c)
        if not (aire_min < aire < aire_max):
            continue
        
        f = calculer_features_contour(c)
        if not f:
            continue
        
        if f["rapport_axes"] > 5.0:
            continue
        if f["convexite"] < 0.30 and aire < aire_min * 3:
            continue
        
        candidats.append((c, f))
    
    if not candidats:
        return MasseResult(detectee=False, forme=None, contour=None, densite=None,
                          nb_masses=0, score_suspicion=0.0, confiance="haute")
    
    masse_c, masse_f = max(candidats, key=lambda x: x[1]["aire"])
    forme = classer_forme(masse_f)
    contour = classer_contour_masse(enhanced, masse_c, masse_f)
    densite, _ = classer_densite_masse(enhanced, masse_c)
    score = calculer_score_suspicion(forme, contour, densite)
    confiance = "haute" if len(candidats) == 1 else "moyenne"
    
    return MasseResult(detectee=True, forme=forme, contour=contour, densite=densite,
                      nb_masses=len(candidats), score_suspicion=round(score, 2), confiance=confiance)

def detecter_calcifications(img_gray: np.ndarray) -> CalcificationResult:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    
    se_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, se_tophat)
    
    seuil_tophat = np.percentile(tophat, 97) if tophat.size > 0 else 30
    seuil_tophat = max(seuil_tophat, 20)
    
    _, thresh_tophat = cv2.threshold(tophat, seuil_tophat, 255, cv2.THRESH_BINARY)
    
    kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh_tophat, cv2.MORPH_OPEN, kernel_s, iterations=1)
    
    contours_c, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    calcifs_valides = []
    for c in contours_c:
        aire = cv2.contourArea(c)
        if not (3 <= aire <= 120):
            continue
        
        if len(c) >= 5:
            try:
                ell = cv2.fitEllipse(c)
                axes = ell[1]
                elong = max(axes) / max(min(axes), 0.1)
                if elong > 4.0:
                    continue
            except:
                pass
        
        calcifs_valides.append(c)
    
    nb_calcif = len(calcifs_valides)
    
    if nb_calcif < 3:
        return CalcificationResult(detectee=False, forme=None, repartition=None,
                                   benigne=False, nb_calcif=0, confiance="haute")
    
    benigne = nb_calcif > 10
    confiance = "haute" if nb_calcif > 10 else "moyenne"
    
    return CalcificationResult(detectee=True, forme="punctiformes_irregulieres",
                               repartition="eparses", benigne=benigne,
                               nb_calcif=nb_calcif, confiance=confiance)

def draw_lesions_on_image(img_color: np.ndarray, img_gray: np.ndarray) -> np.ndarray:
    """Dessine les lésions sur l'image couleur originale (RGB)."""
    img_result = img_color.copy()
    
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    
    h, w = img_gray.shape
    aire_min = int((min(h, w) * 0.03) ** 2)
    aire_max = int((min(h, w) * 0.45) ** 2)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    
    sigma1, sigma2 = 2.0, 20.0
    g1 = cv2.GaussianBlur(enhanced, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(enhanced, (0, 0), sigma2)
    dog = cv2.subtract(g1, g2)
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    block = max(w // 8, 51) | 1
    thresh_adapt = cv2.adaptiveThreshold(dog_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, block, -5)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(thresh_adapt, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        aire = cv2.contourArea(c)
        if aire_min < aire < aire_max:
            if len(c) >= 5:
                try:
                    ell = cv2.fitEllipse(c)
                    axes = ell[1]
                    elong = max(axes) / max(min(axes), 0.1)
                    if elong < 5.0:
                        cv2.drawContours(img_result, [c], -1, (0, 255, 0), 2)
                except:
                    cv2.drawContours(img_result, [c], -1, (0, 255, 0), 2)
    
    se_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, se_tophat)
    seuil_tophat = np.percentile(tophat, 97) if tophat.size > 0 else 30
    seuil_tophat = max(seuil_tophat, 20)
    _, thresh_tophat = cv2.threshold(tophat, seuil_tophat, 255, cv2.THRESH_BINARY)
    
    kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_c = cv2.morphologyEx(thresh_tophat, cv2.MORPH_OPEN, kernel_s, iterations=1)
    contours_c, _ = cv2.findContours(cleaned_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours_c:
        aire = cv2.contourArea(c)
        if 3 <= aire <= 120:
            cv2.drawContours(img_result, [c], -1, (0, 255, 255), 1)
    
    return img_result

def detect_lesions_with_visualization(img_array: np.ndarray) -> tuple:
    """Détecte les lésions et retourne le rapport BI-RADS et l'image annotée."""
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_color = img_array
    else:
        img_gray = img_array
        img_color = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    masse = detecter_masses(img_gray)
    calcification = detecter_calcifications(img_gray)
    
    if masse.detectee and calcification.detectee:
        conclusion = "masse_et_calcification"
    elif masse.detectee:
        conclusion = "masse"
    elif calcification.detectee:
        conclusion = "calcification"
    else:
        conclusion = "normal"
    
    score = masse.score_suspicion if masse.detectee else 0.0
    
    if score >= 0.70:
        suspicion = "tres_suspect"
        birads = "BI-RADS 4"
    elif score >= 0.40:
        suspicion = "suspect"
        birads = "BI-RADS 4"
    elif score >= 0.15 or (calcification.detectee and calcification.benigne):
        suspicion = "probablement_benin"
        birads = "BI-RADS 3"
    else:
        suspicion = "benin"
        birads = "BI-RADS 2"
    
    # Si aucune lésion détectée
    if conclusion == "normal":
        birads = "BI-RADS 1"
    
    rapport = LesionReport(
        fichier="",
        masse=masse,
        calcification=calcification,
        conclusion=conclusion,
        suspicion=suspicion,
        birads=birads,
        recommandation=BIRADS_RECO.get(birads, "")
    )
    
    img_annotated = draw_lesions_on_image(img_color, img_gray)
    
    return rapport, img_annotated

# ==========================================================
# ANALYSE D'UNE IMAGE
# ==========================================================

def analyze_image(img, lr_model, density_model, threshold_lr, use_density):
    results = {"success": True}
    try:
        img_array_preprocessed = preprocess_pil_image(img, IMG_SIZE_PREPROC)
        results["preprocessed"] = img_array_preprocessed
        
        if lr_model:
            arr_lr = prepare_for_model(img_array_preprocessed, IMG_SIZE_LR)
            label, conf, raw = predict_laterality(lr_model, arr_lr, threshold_lr)
            results.update({
                "lr_label":      label,
                "lr_confidence": conf,
                "lr_raw_prob":   raw
            })

        if use_density and density_model:
            arr_density = prepare_for_model(img_array_preprocessed, IMG_SIZE_DENSITY)
            label_d, conf_d, probs_d = predict_density(density_model, arr_density)
            results.update({
                "density_label":      label_d,
                "density_confidence": conf_d,
                "density_probs":      probs_d
            })

    except Exception as e:
        results = {"success": False, "error": str(e)}

    return results

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Analyse Mammaire Complète",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Analyse Mammaire Complète")
st.caption("Traitement d'image · Densité ACR · Lésions BI-RADS")
st.markdown("---")

# ==========================================================
# CHARGEMENT DES MODÈLES
# ==========================================================

with st.spinner("Chargement des modèles…"):
    lr_model,      lr_error      = load_laterality_model()
    density_model, density_error = load_density_model()

col_s1, col_s2 = st.columns(2)
with col_s1:
    if lr_model:
        st.success("✅ Modèle latéralité chargé")
    else:
        st.error(f"❌ Modèle latéralité : {lr_error}")
with col_s2:
    if density_model:
        st.success("✅ Modèle densité chargé")
    else:
        st.error(f"❌ Modèle densité : {density_error}")

if not lr_model and not density_model:
    st.error("Aucun modèle disponible. Vérifiez les fichiers .keras.")
    st.stop()

# ==========================================================
# CONFIGURATION
# ==========================================================

st.markdown("---")
st.subheader("⚙️ Configuration")

col_c1, col_c2, col_c3 = st.columns(3)

with col_c1:
    threshold_lr = st.slider(
        "Seuil de décision — Latéralité",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.01
    )

with col_c2:
    use_density = st.checkbox(
        "Activer l'analyse de densité ACR",
        value=density_model is not None,
        disabled=density_model is None
    )

with col_c3:
    use_lesion = st.checkbox(
        "🔬 Activer la détection de lésions BI-RADS",
        value=True
    )

# ==========================================================
# UPLOAD
# ==========================================================

st.markdown("---")
uploaded_files = st.file_uploader(
    "📤 Uploader une ou plusieurs images mammaires",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👆 Uploadez une ou plusieurs images pour commencer.")
    with st.expander("ℹ️ Guide d'utilisation"):
        st.markdown("""
        **Pipeline d'analyse en 3 étapes :**
        
        1. **🖼️ Traitement d'image** — Prétraitement CLAHE + sharpening
        2. **📊 Densité ACR** — Classification B / C / D
        3. **🔬 Lésions BI-RADS** — Détection masses et calcifications
        
        **BI-RADS Classification :**
        - **BI-RADS 1** : Mammographie normale
        - **BI-RADS 2** : Anomalie bénigne
        - **BI-RADS 3** : Anomalie probablement bénigne
        - **BI-RADS 4** : Anomalie suspecte
        """)
    st.stop()

st.success(f"{len(uploaded_files)} image(s) chargée(s)")

# ==========================================================
# BOUTONS D'ACTION
# ==========================================================

if "results" not in st.session_state:
    st.session_state.results = {}
    
if "lesion_results" not in st.session_state:
    st.session_state.lesion_results = {}
    
if "annotated_images" not in st.session_state:
    st.session_state.annotated_images = {}

col_b1, col_b2, col_b3 = st.columns(3)
with col_b1:
    run = st.button(
        "🔍 Analyser toutes les images",
        type="primary",
        use_container_width=True
    )
with col_b2:
    export = st.button("📊 Exporter CSV", use_container_width=True)
with col_b3:
    if st.button("🗑️ Effacer", use_container_width=True):
        st.session_state.results = {}
        st.session_state.lesion_results = {}
        st.session_state.annotated_images = {}
        st.rerun()

# ==========================================================
# ANALYSE
# ==========================================================

if run:
    bar = st.progress(0)
    info = st.empty()
    for i, f in enumerate(uploaded_files):
        info.text(f"Analyse de {f.name}… ({i+1}/{len(uploaded_files)})")
        img = Image.open(f)
        img_array = np.array(img)
        
        st.session_state.results[f.name] = analyze_image(
            img, lr_model, density_model, threshold_lr, use_density
        )
        
        if use_lesion and st.session_state.results[f.name].get("success"):
            try:
                lesion_rapport, img_annotated = detect_lesions_with_visualization(img_array)
                lesion_rapport.fichier = f.name
                st.session_state.lesion_results[f.name] = lesion_rapport
                st.session_state.annotated_images[f.name] = img_annotated
            except Exception as e:
                st.session_state.lesion_results[f.name] = {"error": str(e)}
        
        bar.progress((i + 1) / len(uploaded_files))
    bar.empty()
    info.empty()
    st.success("✅ Analyse terminée !")

# ==========================================================
# EXPORT CSV
# ==========================================================

if export and st.session_state.results:
    rows = []
    for fname, res in st.session_state.results.items():
        row = {"Fichier": fname}
        if res.get("success"):
            row["Statut"] = "OK"
            row["Latéralité"] = res.get("lr_label", "—")
            row["Confiance_LR_%"] = f"{res.get('lr_confidence', 0)*100:.1f}"
            
            if use_density and "density_label" in res:
                row["Densité"] = res["density_label"]
                row["Confiance_ACR_%"] = f"{res.get('density_confidence', 0)*100:.1f}"
            
            lesion = st.session_state.lesion_results.get(fname)
            if lesion and hasattr(lesion, 'masse'):
                row["BI-RADS"] = lesion.birads
                row["Recommandation"] = lesion.recommandation
                row["Masse_détectée"] = "Oui" if lesion.masse.detectee else "Non"
                row["Calcif_détectée"] = "Oui" if lesion.calcification.detectee else "Non"
                row["Suspicion"] = lesion.suspicion
        else:
            row["Statut"] = "Erreur"
            row["Erreur"] = res.get("error", "")
        rows.append(row)
    
    csv = pd.DataFrame(rows).to_csv(index=False)
    st.download_button(
        "📥 Télécharger le CSV",
        data=csv,
        file_name="analyse_mammaire_complete.csv",
        mime="text/csv",
        use_container_width=True
    )

# ==========================================================
# AFFICHAGE DES RÉSULTATS EN 3 COLONNES
# ==========================================================

st.markdown("---")

# Sélection du mode d'affichage
view = st.radio(
    "Mode d'affichage",
    ["Grille", "Détail", "Tableau", "Comparaison 3 colonnes"],
    horizontal=True
)

# ==========================================================
# HELPERS D'AFFICHAGE
# ==========================================================

def density_badge(res):
    label = res.get("density_label", "—")
    conf  = res.get("density_confidence", 0) * 100
    color = DENSITY_COLORS.get(label, "#999")
    desc  = DENSITY_DESC.get(label, "")
    return (
        f'<div style="background:{color};padding:8px 12px;border-radius:6px;'
        f'color:#fff;text-align:center;margin:4px 0">'
        f'<strong>{label}</strong><br>'
        f'<small>{desc}</small><br>'
        f'<small>Confiance : {conf:.1f}%</small></div>'
    )

def birads_badge(birads):
    color = COULEURS_BIRADS.get(birads, "#999")
    return f'<span style="color:{color};font-weight:bold;font-size:1.2rem">{birads}</span>'

def lesion_summary(lesion):
    if not lesion or not hasattr(lesion, 'masse'):
        return ""
    
    suspicion_colors = {
        "benin": "#4CAF50",
        "probablement_benin": "#8BC34A",
        "suspect": "#FF9800",
        "tres_suspect": "#F44336"
    }
    color = suspicion_colors.get(lesion.suspicion, "#999")
    
    suspicion_fr = {
        "benin": "Bénin",
        "probablement_benin": "Probablement bénin",
        "suspect": "Suspect",
        "tres_suspect": "Très suspect"
    }.get(lesion.suspicion, lesion.suspicion)
    
    masse_info = ""
    if lesion.masse.detectee:
        masse_info = f"Masse: {lesion.masse.forme} | {lesion.masse.contour}<br>"
    if lesion.calcification.detectee:
        masse_info += f"Calcif: {lesion.calcification.nb_calcif}"
    
    return (
        f'<div style="border-left:4px solid {color};padding:8px 12px;margin:4px 0">'
        f'<strong style="color:{color}">{lesion.birads}</strong><br>'
        f'<small>{suspicion_fr}</small><br>'
        f'<small>{masse_info}</small></div>'
    )

def display_three_columns(original, preprocessed, annotated, res, lesion):
    """Affiche les résultats en 3 colonnes : Traitement | Densité | Lésions"""
    
    col1, col2, col3 = st.columns(3, gap="medium")
    
    # Colonne 1: Traitement d'image
    with col1:
        st.markdown("### 🖼️ Traitement d'image")
        
        # Original
        st.caption("📷 **Original**")
        st.image(original, use_container_width=True)
        
        # Prétraitée
        if preprocessed is not None:
            st.caption("🔧 **Prétraitée**")
            st.image(preprocessed, use_container_width=True, clamp=True)
        
        # Latéralité
        if res and res.get("lr_label"):
            lr_label = res.get("lr_label")
            lr_conf = res.get("lr_confidence", 0) * 100
            icon = "◀" if lr_label == "Gauche" else "▶"
            st.markdown(f"**Latéralité:** {icon} {lr_label} ({lr_conf:.1f}%)")
    
    # Colonne 2: Densité ACR
    with col2:
        st.markdown("### 📊 Densité ACR")
        
        if res and "density_label" in res:
            st.markdown(density_badge(res), unsafe_allow_html=True)
            
            # Probabilités détaillées
            probs = res.get("density_probs", {})
            st.markdown("**Probabilités:**")
            for cls in ["ACR_B", "ACR_C", "ACR_D"]:
                prob = probs.get(cls, 0) * 100
                color = DENSITY_COLORS.get(cls, "#999")
                st.markdown(
                    f'<div style="margin:4px 0">'
                    f'<span style="font-weight:bold;color:{color}">{cls}</span>'
                    f'<div style="background:#e0e0e0;border-radius:4px;height:16px;width:100%;'
                    f'position:relative;margin-top:2px">'
                    f'<div style="background:{color};width:{prob:.1f}%;height:16px;'
                    f'border-radius:4px;text-align:right;padding-right:4px;'
                    f'color:white;font-size:10px;line-height:16px">'
                    f'{prob:.1f}%</div></div></div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Analyse de densité non activée")
    
    # Colonne 3: Lésions BI-RADS
    with col3:
        st.markdown("### 🔬 Lésions BI-RADS")
        
        if lesion and hasattr(lesion, 'masse'):
            # Badge BI-RADS
            st.markdown(f"**Classification:** {birads_badge(lesion.birads)}", unsafe_allow_html=True)
            
            # Résumé
            st.markdown(lesion_summary(lesion), unsafe_allow_html=True)
            
            # Recommandation
            if lesion.recommandation:
                st.info(lesion.recommandation)
            
            # Détails
            if lesion.masse.detectee:
                with st.expander("📋 Détails masse"):
                    st.write(f"- **Forme:** {lesion.masse.forme}")
                    st.write(f"- **Contour:** {lesion.masse.contour}")
                    st.write(f"- **Densité:** {lesion.masse.densite}")
                    st.write(f"- **Score suspicion:** {lesion.masse.score_suspicion:.2f}/1.0")
            
            if lesion.calcification.detectee:
                with st.expander("📋 Détails calcifications"):
                    st.write(f"- **Nombre:** {lesion.calcification.nb_calcif}")
                    st.write(f"- **Forme:** {lesion.calcification.forme}")
                    st.write(f"- **Répartition:** {lesion.calcification.repartition}")
            
            # Image annotée
            if annotated is not None:
                st.caption("🔍 **Image annotée** (Vert: masses | Jaune: calcifications)")
                st.image(annotated, use_container_width=True)
        else:
            st.info("Détection de lésions non activée ou non effectuée")

# ==========================================================
# AFFICHAGE PAR MODE
# ==========================================================

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
                
                res = st.session_state.results.get(f.name)
                lesion = st.session_state.lesion_results.get(f.name)
                annotated = st.session_state.annotated_images.get(f.name)
                
                # Affichage en 3 colonnes compact
                display_three_columns(
                    Image.open(f),
                    res.get("preprocessed") if res else None,
                    annotated,
                    res,
                    lesion
                )

# ---------- DÉTAIL ----------
elif view == "Détail":
    for f in uploaded_files:
        with st.expander(f"📷 {f.name}"):
            res = st.session_state.results.get(f.name)
            lesion = st.session_state.lesion_results.get(f.name)
            annotated = st.session_state.annotated_images.get(f.name)
            
            if res and res.get("success"):
                display_three_columns(
                    Image.open(f),
                    res.get("preprocessed"),
                    annotated,
                    res,
                    lesion
                )
            else:
                st.error(res.get("error", "Erreur inconnue") if res else "Non analysé")

# ---------- TABLEAU ----------
elif view == "Tableau":
    rows = []
    for f in uploaded_files:
        res = st.session_state.results.get(f.name, {})
        lesion = st.session_state.lesion_results.get(f.name)
        
        row = {"Fichier": f.name}
        if res.get("success"):
            row["Latéralité"] = res.get("lr_label", "—")
            row["Conf. LR"] = f"{res.get('lr_confidence', 0)*100:.1f}%"
            if use_density:
                row["Densité"] = res.get("density_label", "—")
                row["Conf. ACR"] = f"{res.get('density_confidence', 0)*100:.1f}%"
            if use_lesion and lesion and hasattr(lesion, 'masse'):
                row["BI-RADS"] = lesion.birads
                row["Masse"] = "✓" if lesion.masse.detectee else "✗"
                row["Calcif"] = "✓" if lesion.calcification.detectee else "✗"
                row["Suspicion"] = lesion.suspicion
            row["Statut"] = "✅"
        elif res:
            row["Statut"] = "❌"
        else:
            row["Statut"] = "⏳"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------- COMPARAISON 3 COLONNES ----------
else:  # view == "Comparaison 3 colonnes"
    st.subheader("📊 Comparaison détaillée en 3 colonnes")
    
    if len(uploaded_files) > 0:
        selected_file = st.selectbox(
            "Sélectionner une image pour comparaison détaillée",
            [f.name for f in uploaded_files],
            index=0
        )
        
        f = next((f for f in uploaded_files if f.name == selected_file), None)
        if f:
            res = st.session_state.results.get(f.name)
            lesion = st.session_state.lesion_results.get(f.name)
            annotated = st.session_state.annotated_images.get(f.name)
            
            if res and res.get("success"):
                display_three_columns(
                    Image.open(f),
                    res.get("preprocessed"),
                    annotated,
                    res,
                    lesion
                )
            else:
                st.info("Cette image n'a pas encore été analysée.")

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
        st.markdown("**Latéralité**")
        gauche = sum(1 for n in analyzed 
                    if st.session_state.results[n].get("lr_label") == "Gauche")
        droite = len(analyzed) - gauche
        st.markdown(f"◀ Gauche : **{gauche}**")
        st.markdown(f"Droite ▶ : **{droite}**")
        
        if use_density:
            st.markdown("---")
            st.markdown("**Densité ACR**")
            for cls in ["ACR_B", "ACR_C", "ACR_D"]:
                count = sum(1 for n in analyzed 
                           if st.session_state.results[n].get("density_label") == cls)
                if count:
                    color = DENSITY_COLORS[cls]
                    st.markdown(f'<span style="color:{color}">■</span> {cls} : **{count}**',
                               unsafe_allow_html=True)
        
        if use_lesion:
            st.markdown("---")
            st.markdown("**BI-RADS**")
            birads_counts = {}
            for n in analyzed:
                lesion = st.session_state.lesion_results.get(n)
                if lesion and hasattr(lesion, 'birads'):
                    birads_counts[lesion.birads] = birads_counts.get(lesion.birads, 0) + 1
            
            for birads in ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4"]:
                count = birads_counts.get(birads, 0)
                if count:
                    color = COULEURS_BIRADS.get(birads, "#999")
                    st.markdown(f'<span style="color:{color}">■</span> {birads} : **{count}**',
                               unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**Modèles**")
    st.markdown("✅ Latéralité" if lr_model else "❌ Latéralité")
    st.markdown("✅ Densité" if density_model else "❌ Densité")
    st.markdown("✅ Détection BI-RADS")
    st.caption(f"TensorFlow {tf.__version__}")

gc.collect()
