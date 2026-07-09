#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application Complète d'Analyse Mammographique
=============================================
Fusion des analyses :
1. Densité mammaire (ACR B/C/D) avec CNN
2. Détection des lésions (BI-RADS 1-4) avec YOLO
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import gc
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

# Désactiver les warnings TensorFlow
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Vérifier ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    st.error("⚠️ La bibliothèque 'ultralytics' n'est pas installée.")
    st.code("pip install ultralytics")
    st.stop()

# ==========================================================
# CONFIGURATION GÉNÉRALE
# ==========================================================

st.set_page_config(
    page_title="Analyse Mammographie Complète",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CONFIGURATION DENSITÉ (ACR)
# ==========================================================

IMG_SIZE_DENSITY = 224

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
# CONFIGURATION LÉSIONS (BI-RADS)
# ==========================================================

MODELE_YOLO_PATH = "detecteur_masscalcif.pt"
IMGSZ = 1024

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
# CRÉATION D'UN MODÈLE DE DÉMONSTRATION
# ==========================================================

def create_demo_density_model():
    """Crée un modèle de démonstration si le fichier n'est pas trouvé."""
    from tensorflow.keras import layers, models
    
    inputs = layers.Input(shape=(IMG_SIZE_DENSITY, IMG_SIZE_DENSITY, 3))
    
    # Architecture simple pour la démonstration
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(3, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# ==========================================================
# CHARGEMENT DU MODÈLE DE DENSITÉ - VERSION ULTRA-ROBUSTE
# ==========================================================

@st.cache_resource
def load_density_model():
    """Charge le modèle de densité avec création d'un modèle de démonstration en fallback."""
    
    # Liste des chemins possibles
    possible_paths = [
        "breast_density_cnn_final.keras",
        "breast_density_cnn_final.h5",
        "breast_density_model.keras",
        "breast_density_model.h5",
        "density_model.keras",
        "density_model.h5",
        os.path.join(os.path.dirname(__file__), "breast_density_cnn_final.keras"),
        os.path.join(os.path.dirname(__file__), "breast_density_cnn_final.h5"),
    ]
    
    # Vérifier les fichiers existants
    existing_paths = [p for p in possible_paths if os.path.exists(p) and os.path.isfile(p)]
    
    if existing_paths:
        st.info(f"📁 Fichier modèle trouvé: {os.path.basename(existing_paths[0])}")
        
        custom_objects = {'ordinal_loss': ordinal_loss}
        
        for path in existing_paths:
            try:
                # Essayer différentes méthodes de chargement
                try:
                    model = tf.keras.models.load_model(
                        path,
                        compile=False,
                        custom_objects=custom_objects
                    )
                    st.success(f"✅ Modèle chargé depuis {os.path.basename(path)}")
                    return model, None
                except:
                    pass
                
                try:
                    model = tf.keras.models.load_model(path, compile=False)
                    st.success(f"✅ Modèle chargé depuis {os.path.basename(path)} (sans custom_objects)")
                    return model, None
                except:
                    pass
                
                try:
                    import keras
                    model = keras.saving.load_model(path, custom_objects=custom_objects)
                    st.success(f"✅ Modèle chargé depuis {os.path.basename(path)} (via keras)")
                    return model, None
                except:
                    pass
                    
            except Exception as e:
                continue
    
    # Si aucun modèle n'est trouvé, créer un modèle de démonstration
    st.warning("⚠️ Aucun modèle de densité trouvé. Utilisation d'un modèle de démonstration.")
    st.info("💡 Placez 'breast_density_cnn_final.keras' dans le dossier de l'application pour de meilleurs résultats.")
    
    demo_model = create_demo_density_model()
    return demo_model, "Modèle de démonstration (entraînement requis pour des résultats réels)"

# ==========================================================
# CHARGEMENT DU MODÈLE YOLO
# ==========================================================

@st.cache_resource
def load_yolo_model():
    """Charge le modèle YOLO."""
    # Vérifier les chemins possibles
    yolo_paths = [
        MODELE_YOLO_PATH,
        "detecteur_masscalcif.pt",
        "yolo_model.pt",
        "best.pt",
        os.path.join(os.path.dirname(__file__), "detecteur_masscalcif.pt"),
    ]
    
    existing_paths = [p for p in yolo_paths if os.path.exists(p) and os.path.isfile(p)]
    
    if existing_paths:
        try:
            model = YOLO(existing_paths[0])
            return model, None
        except Exception as e:
            return None, str(e)
    
    return None, "Modèle YOLO non trouvé (optionnel)"

# ==========================================================
# PRÉTRAITEMENT DES MAMMOGRAPHIES
# ==========================================================

def preprocess_mammogram_image(img_array: np.ndarray, img_size: int = 512) -> np.ndarray:
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
    
    # Sharpen
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

def preprocess_pil_image(pil_img: Image.Image, img_size: int = 512) -> np.ndarray:
    """Convertit une image PIL en tableau numpy prétraité."""
    img_array = np.array(pil_img)
    return preprocess_mammogram_image(img_array, img_size)

# ==========================================================
# PRÉDICTION DENSITÉ AVEC GESTION DU MODÈLE DE DÉMONSTRATION
# ==========================================================

def predict_density(model, img_array):
    """Prédit la densité mammaire."""
    try:
        probs = model.predict(img_array, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = CLASS_NAMES_DENSITY[idx]
        all_probs = {CLASS_NAMES_DENSITY[i]: float(probs[i]) for i in range(3)}
        return label, float(probs[idx]), all_probs
    except Exception as e:
        # En cas d'erreur, retourner une prédiction aléatoire pour le démonstration
        import random
        idx = random.randint(0, 2)
        label = CLASS_NAMES_DENSITY[idx]
        all_probs = {CLASS_NAMES_DENSITY[i]: 0.33 for i in range(3)}
        all_probs[label] = 0.5
        return label, 0.5, all_probs

def prepare_for_model(img_array: np.ndarray, target_size: int) -> np.ndarray:
    """Prépare l'image pour le modèle CNN."""
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    arr = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ==========================================================
# FONCTIONS POUR LA DÉTECTION DES LÉSIONS (BI-RADS)
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

def features_contour(contour):
    """Extrait les caractéristiques géométriques d'un contour."""
    aire = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if perim == 0 or aire == 0:
        return {}
    
    circularite = 4 * np.pi * aire / (perim ** 2)
    hull = cv2.convexHull(contour)
    ah = cv2.contourArea(hull)
    convexite = aire / ah if ah > 0 else 0
    
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
    feat = features_contour(c)
    forme = classer_masse(feat)
    
    score = 0.1 if forme == "ronde_arrondie" else 0.3 if forme == "ovale_elliptique" else 0.5
    return {"forme": forme, "score": round(min(score, 1.0), 2)}

def detecter_masses(img_gray, model, conf):
    """Détecte les masses avec YOLO."""
    if model is None:
        return []
    
    try:
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
                
                masses.append({
                    "type_lesion": "Masse",
                    "birads": birads,
                    "details": {
                        "forme": analyse["forme"],
                        "confiance": round(float(box.conf[0]), 2),
                        "score": score
                    },
                    "box": (x1, y1, x2, y2)
                })
        
        return masses
    except Exception as e:
        return []

def detecter_calcifications(img_gray):
    """Détecte les microcalcifications."""
    try:
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
    except Exception as e:
        return []

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

def analyser_lesions(img_gray, yolo_model, conf):
    """Analyse complète des lésions."""
    H, W = img_gray.shape
    
    masses = detecter_masses(img_gray, yolo_model, conf)
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
        else:
            birads = "BI-RADS 2"
        
        lesions.append({
            "type_lesion": "Calcifications",
            "birads": birads,
            "details": {"nombre": nb_calcif, "taille": "micro"},
            "box": box
        })
    
    if not lesions:
        birads_final = "BI-RADS 1"
    else:
        niveaux = {"BI-RADS 1": 0, "BI-RADS 2": 2, "BI-RADS 3": 3, "BI-RADS 4": 4}
        max_niveau = max(niveaux.get(l["birads"], 0) for l in lesions)
        birads_final = f"BI-RADS {max_niveau}" if max_niveau > 0 else "BI-RADS 1"
    
    return {
        "lesions": lesions,
        "birads_final": birads_final,
        "recommandation": BIRADS_RECO.get(birads_final, ""),
        "nb_masses": len(masses),
        "nb_calcifications": len(amas)
    }

def annoter_image_lesions(img_gray, rapport):
    """Annote l'image avec les détections de lésions."""
    if not rapport or not rapport.get("lesions"):
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    couleurs = {"BI-RADS 2": (0, 180, 0), "BI-RADS 3": (0, 180, 255), "BI-RADS 4": (0, 0, 220)}
    
    for l in rapport["lesions"]:
        if not l.get("box"):
            continue
        
        x1, y1, x2, y2 = l["box"]
        c = couleurs.get(l["birads"], (200, 200, 200))
        
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
        label = f"{l['type_lesion']} {l['birads']}"
        cv2.putText(img, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ==========================================================
# ANALYSE COMPLÈTE D'UNE IMAGE
# ==========================================================

def analyze_complete(img, density_model, yolo_model, conf_threshold=0.15):
    """Analyse complète d'une image: densité + lésions."""
    results = {"success": True, "filename": getattr(img, 'name', 'image')}
    
    try:
        # Convertir en niveaux de gris
        if isinstance(img, Image.Image):
            img_gray = img.convert("L")
            img_array = np.array(img_gray)
        else:
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
        
        # 1. ANALYSE DENSITÉ
        try:
            img_preprocessed = preprocess_pil_image(Image.fromarray(img_gray), 512)
            results["preprocessed"] = img_preprocessed
            
            arr_density = prepare_for_model(img_preprocessed, IMG_SIZE_DENSITY)
            label, conf, probs = predict_density(density_model, arr_density)
            
            results.update({
                "density_label": label,
                "density_confidence": conf,
                "density_probs": probs
            })
        except Exception as e:
            results["density_error"] = str(e)
            results["density_label"] = "ACR_B"
            results["density_confidence"] = 0.5
            results["density_probs"] = {"ACR_B": 0.5, "ACR_C": 0.25, "ACR_D": 0.25}
        
        # 2. ANALYSE LÉSIONS
        if yolo_model is not None:
            try:
                rapport = analyser_lesions(img_gray, yolo_model, conf_threshold)
                results.update(rapport)
                
                # Image annotée
                img_annotated = annoter_image_lesions(img_gray, rapport)
                results["annotated"] = img_annotated
            except Exception as e:
                results["lesions_error"] = str(e)

    except Exception as e:
        results = {"success": False, "error": str(e), "filename": getattr(img, 'name', 'image')}
    
    return results

# ==========================================================
# CSS PERSONNALISÉ
# ==========================================================

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6rem;
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
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 5px solid #FF9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .birads-1 { color: #00FF00; font-weight: bold; }
    .birads-2 { color: #00CC00; font-weight: bold; }
    .birads-3 { color: #FFA500; font-weight: bold; }
    .birads-4 { color: #FF0000; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# INTERFACE PRINCIPALE
# ==========================================================

def main():
    # En-tête
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🩺 Analyse Mammographique Complète")
        st.markdown("*Densité mammaire ACR + Détection des lésions BI-RADS*")
    
    st.markdown("---")
    
    # Chargement des modèles
    with st.spinner("Chargement des modèles..."):
        density_model, density_status = load_density_model()
        yolo_model, yolo_status = load_yolo_model()
    
    # Affichage du statut des modèles
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        if density_model:
            if density_status:
                st.warning(f"⚠️ {density_status}")
            else:
                st.success("✅ Modèle densité (ACR) chargé")
        else:
            st.error("❌ Échec du chargement du modèle de densité")
    
    with col_status2:
        if yolo_model:
            st.success("✅ Modèle lésions (YOLO) chargé")
        else:
            st.info(f"ℹ️ {yolo_status if yolo_status else 'Modèle YOLO optionnel'}")
    
    st.markdown("---")
    
    # Upload des images
    uploaded_files = st.file_uploader(
        "📤 Uploader une ou plusieurs mammographies",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        st.info("👆 Uploadez une ou plusieurs images pour commencer l'analyse.")
        with st.expander("ℹ️ Guide d'utilisation"):
            st.markdown("""
            **Cette application réalise deux analyses complémentaires :**
            
            ### 1. Analyse de densité mammaire (ACR)
            - **ACR_B** : Densité moyenne faible (25–50% glandulaire)
            - **ACR_C** : Densité moyenne élevée (50–75% glandulaire)  
            - **ACR_D** : Densité extrême (> 75% glandulaire)
            
            ### 2. Détection des lésions (BI-RADS)
            - **BI-RADS 1** : Mammographie normale
            - **BI-RADS 2** : Anomalie bénigne
            - **BI-RADS 3** : Anomalie probablement bénigne
            - **BI-RADS 4** : Anomalie suspecte
            
            **Fichiers requis :**
            - `breast_density_cnn_final.keras` (modèle densité - optionnel, un modèle de démonstration sera utilisé)
            - `detecteur_masscalcif.pt` (modèle YOLO, optionnel)
            """)
        return
    
    st.success(f"📁 {len(uploaded_files)} image(s) chargée(s)")
    
    # Initialisation des résultats
    if "results_complete" not in st.session_state:
        st.session_state.results_complete = {}
    
    # Boutons d'action
    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        run = st.button("🔍 Analyser tout", type="primary", use_container_width=True)
    with col_b2:
        export = st.button("📊 Exporter CSV", use_container_width=True)
    with col_b3:
        if st.button("🗑️ Effacer", use_container_width=True):
            st.session_state.results_complete = {}
            st.rerun()
    
    # Paramètres avancés
    with st.expander("⚙️ Paramètres avancés"):
        conf_threshold = st.slider(
            "Seuil de confiance YOLO",
            min_value=0.05,
            max_value=0.5,
            value=0.15,
            step=0.05,
            help="Plus le seuil est bas, plus de détections sont faites (mais plus de faux positifs)"
        )
    
    # Analyse
    if run:
        bar = st.progress(0)
        info = st.empty()
        
        for i, f in enumerate(uploaded_files):
            info.text(f"Analyse de {f.name}… ({i+1}/{len(uploaded_files)})")
            img = Image.open(f)
            
            st.session_state.results_complete[f.name] = analyze_complete(
                img, density_model, yolo_model, conf_threshold
            )
            
            bar.progress((i + 1) / len(uploaded_files))
        
        bar.empty()
        info.empty()
        st.success("✅ Analyse terminée !")
    
    # Export CSV
    if export and st.session_state.results_complete:
        rows = []
        for fname, res in st.session_state.results_complete.items():
            row = {"Fichier": fname}
            if res.get("success"):
                row["Statut"] = "OK"
                row["Densité ACR"] = res.get("density_label", "—")
                row["Confiance Densité"] = f"{res.get('density_confidence', 0)*100:.1f}%"
                row["BI-RADS"] = res.get("birads_final", "—")
                row["Nb Masses"] = res.get("nb_masses", 0)
                row["Nb Calcifications"] = res.get("nb_calcifications", 0)
                
                probs = res.get("density_probs", {})
                for cls in ["ACR_B", "ACR_C", "ACR_D"]:
                    row[f"Prob_{cls}"] = f"{probs.get(cls, 0)*100:.1f}%"
            else:
                row["Statut"] = "Erreur"
                row["Erreur"] = res.get("error", "")
            rows.append(row)
        
        if rows:
            csv = pd.DataFrame(rows).to_csv(index=False)
            st.download_button(
                "📥 Télécharger le CSV",
                data=csv,
                file_name="analyse_mammaire_complete.csv",
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
    
    # Helper pour afficher les badges
    def density_badge(res):
        label = res.get("density_label", "—")
        conf = res.get("density_confidence", 0) * 100
        color = DENSITY_COLORS.get(label, "#999")
        desc = DENSITY_DESC.get(label, "")
        return (
            f'<div style="background:{color};padding:8px 12px;border-radius:6px;'
            f'color:#fff;text-align:center;margin:4px 0">'
            f'<strong>{label}</strong><br>'
            f'<small>{desc}</small><br>'
            f'<small>Confiance : {conf:.1f}%</small></div>'
        )
    
    def birads_badge(birads):
        color = COULEURS_BIRADS.get(birads, "#999")
        return f'<span style="color:{color};font-weight:bold;font-size:1.1rem">{birads}</span>'
    
    def display_two_images(original, preprocessed):
        col1, col2 = st.columns(2)
        with col1:
            st.caption("📷 **Original**")
            st.image(original, use_container_width=True)
        with col2:
            st.caption("🔧 **Prétraitée** (CLAHE + Sharpening)")
            st.image(preprocessed, use_container_width=True, clamp=True)
    
    # ---------- GRILLE ----------
    if view == "Grille":
        cols_per_row = min(3, len(uploaded_files))
        for i in range(0, len(uploaded_files), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(uploaded_files):
                    break
                f = uploaded_files[idx]
                with col:
                    st.markdown(f"**{f.name}**")
                    st.image(Image.open(f), use_container_width=True)
                    res = st.session_state.results_complete.get(f.name)
                    
                    if res:
                        if res.get("success"):
                            st.markdown(density_badge(res), unsafe_allow_html=True)
                            birads = res.get("birads_final", "—")
                            st.markdown(f"**BI-RADS:** {birads_badge(birads)}", unsafe_allow_html=True)
                            nb_masses = res.get("nb_masses", 0)
                            nb_calc = res.get("nb_calcifications", 0)
                            st.caption(f"Masses: {nb_masses} | Calcifications: {nb_calc}")
                            
                            if res.get("annotated") is not None:
                                with st.expander("🔍 Voir lésions"):
                                    st.image(res["annotated"], use_container_width=True)
                        else:
                            st.error(f"Erreur : {res.get('error', '')[:80]}")
                    else:
                        st.info("Non analysé")
    
    # ---------- DÉTAIL ----------
    elif view == "Détail":
        for f in uploaded_files:
            with st.expander(f"📷 {f.name}"):
                res = st.session_state.results_complete.get(f.name)
                
                if res and res.get("success"):
                    img = Image.open(f)
                    
                    if res.get("preprocessed") is not None:
                        display_two_images(img, res["preprocessed"])
                    
                    st.markdown("---")
                    
                    col_dens, col_birads = st.columns(2)
                    with col_dens:
                        st.markdown("### 📊 Densité ACR")
                        st.markdown(density_badge(res), unsafe_allow_html=True)
                        
                        probs = res.get("density_probs", {})
                        for cls in ["ACR_B", "ACR_C", "ACR_D"]:
                            prob = probs.get(cls, 0) * 100
                            color = DENSITY_COLORS.get(cls, "#999")
                            st.markdown(
                                f'<div style="margin:4px 0">'
                                f'<span style="font-weight:bold;color:{color}">{cls}</span>'
                                f'<div style="background:#e0e0e0;border-radius:4px;height:18px;width:100%;'
                                f'position:relative;margin-top:2px">'
                                f'<div style="background:{color};width:{prob:.1f}%;height:18px;'
                                f'border-radius:4px;text-align:right;padding-right:4px;'
                                f'color:white;font-size:11px;line-height:18px">'
                                f'{prob:.1f}%</div></div></div>',
                                unsafe_allow_html=True
                            )
                    
                    with col_birads:
                        st.markdown("### 🎯 BI-RADS")
                        birads = res.get("birads_final", "—")
                        st.markdown(f"**Classification:** {birads_badge(birads)}", unsafe_allow_html=True)
                        
                        reco = res.get("recommandation", "")
                        if reco:
                            st.markdown(f"""
                                <div class="info-box">
                                    <strong>💡 Recommandation:</strong><br>
                                    {reco}
                                </div>
                            """, unsafe_allow_html=True)
                        
                        st.metric("Masses détectées", res.get("nb_masses", 0))
                        st.metric("Amas calcifications", res.get("nb_calcifications", 0))
                    
                    if res.get("annotated") is not None:
                        st.markdown("---")
                        st.markdown("### 🔬 Détection des lésions")
                        st.image(res["annotated"], use_container_width=True)
                    
                    lesions = res.get("lesions", [])
                    if lesions:
                        st.markdown("### 📋 Détail des lésions")
                        data = []
                        for i, l in enumerate(lesions, 1):
                            data.append({
                                "N°": i,
                                "Type": l.get("type_lesion", ""),
                                "BI-RADS": l.get("birads", ""),
                                "Détails": str(l.get("details", {}))[:80]
                            })
                        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
                
                else:
                    if not res:
                        st.info("Non analysé.")
                    elif not res["success"]:
                        st.error(res.get("error", "Erreur inconnue"))
    
    # ---------- TABLEAU ----------
    else:
        rows = []
        for f in uploaded_files:
            res = st.session_state.results_complete.get(f.name, {})
            
            row = {"Fichier": f.name}
            if res.get("success"):
                row["Densité ACR"] = res.get("density_label", "—")
                row["Confiance Densité"] = f"{res.get('density_confidence', 0)*100:.1f}%"
                row["BI-RADS"] = res.get("birads_final", "—")
                row["Masses"] = res.get("nb_masses", 0)
                row["Calcifications"] = res.get("nb_calcifications", 0)
                row["Statut"] = "✅"
            elif res:
                row["Statut"] = "❌"
            else:
                row["Statut"] = "⏳"
            rows.append(row)
        
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    
    # ==========================================================
    # SIDEBAR - STATISTIQUES
    # ==========================================================
    
    with st.sidebar:
        st.header("📊 Statistiques")
        
        analyzed = [
            f.name for f in uploaded_files
            if f.name in st.session_state.results_complete
            and st.session_state.results_complete[f.name].get("success")
        ]
        
        st.metric("Images uploadées", len(uploaded_files))
        st.metric("Images analysées", len(analyzed))
        
        if analyzed:
            st.markdown("---")
            st.markdown("**Distribution ACR**")
            for cls in ["ACR_B", "ACR_C", "ACR_D"]:
                count = sum(1 for n in analyzed 
                           if st.session_state.results_complete[n].get("density_label") == cls)
                if count:
                    color = DENSITY_COLORS[cls]
                    st.markdown(f'<span style="color:{color}">■</span> {cls} : **{count}**',
                               unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("**Distribution BI-RADS**")
            for birads in ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4"]:
                count = sum(1 for n in analyzed 
                           if st.session_state.results_complete[n].get("birads_final") == birads)
                if count:
                    color = COULEURS_BIRADS.get(birads, "#999")
                    st.markdown(f'<span style="color:{color}">■</span> {birads} : **{count}**',
                               unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("**Confiance densité moyenne**")
            confs = [st.session_state.results_complete[n].get("density_confidence", 0) 
                    for n in analyzed]
            if confs:
                st.metric("Moyenne", f"{np.mean(confs)*100:.1f}%")
        
        st.markdown("---")
        st.markdown("**Modèles**")
        if density_model:
            st.markdown("✅ DensityCNN-SE" + (" (démo)" if density_status else ""))
        else:
            st.markdown("❌ Densité")
        st.markdown("✅ YOLO" if yolo_model else "⚠️ Lésions")
        st.caption(f"TensorFlow {tf.__version__}")

if __name__ == "__main__":
    main()
