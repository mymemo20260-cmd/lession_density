#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Mammographie BI-RADS
============================
Interface simple et élégante pour la détection des lésions mammaires.
"""

import cv2
import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import pandas as pd

# ==========================================================
# CONFIGURATION
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

# ==========================================================
# MASQUE DU SEIN
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
# ANALYSE DES MASSES (YOLO)
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
    
    # Score simplifié
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
            
            # Classification BI-RADS simplifiée
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

# ==========================================================
# ANALYSE DES CALCIFICATIONS
# ==========================================================
def detecter_calcifications(img_gray):
    """Détecte les microcalcifications."""
    mask = masque_sein(img_gray)
    
    # Amélioration du contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(img_gray)
    
    # Top-Hat
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    tophat = cv2.morphologyEx(enh, cv2.MORPH_TOPHAT, se)
    
    # Seuillage
    zone = tophat[mask > 0]
    seuil = np.percentile(zone, 99) if zone.size else 30
    _, thr = cv2.threshold(tophat, seuil, 255, cv2.THRESH_BINARY)
    thr = cv2.bitwise_and(thr, thr, mask=mask)
    
    # Nettoyage
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    for c in contours:
        aire = cv2.contourArea(c)
        if 2 <= aire <= 90:  # Taille des microcalcifications
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

# ==========================================================
# PIPELINE PRINCIPAL
# ==========================================================
def analyser(img_gray, model, conf):
    """Pipeline complet d'analyse."""
    H, W = img_gray.shape
    
    # Détection des masses
    masses = detecter_masses(img_gray, model, conf)
    
    # Détection des calcifications
    points = detecter_calcifications(img_gray)
    amas = regrouper_amas(points, W)
    
    # Construction du rapport
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
    
    # Score final
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

def annoter_image(img_gray, rapport):
    """Annote l'image avec les détections."""
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
# INTERFACE STREAMLIT
# ==========================================================
def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Analyse BI-RADS",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # CSS personnalisé pour un look moderne
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
        </style>
    """, unsafe_allow_html=True)
    
    # En-tête
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🩺 Analyse Mammographie")
        st.markdown("*Détection automatique des lésions et classification BI-RADS*")
    
    st.markdown("---")
    
    # Chargement du modèle
    @st.cache_resource
    def load_model():
        try:
            return YOLO(MODELE_PATH)
        except Exception as e:
            st.error(f"⚠️ Erreur de chargement du modèle: {e}")
            return None
    
    model = load_model()
    if model is None:
        st.stop()
    
    # Zone de téléchargement
    uploaded = st.file_uploader(
        "📤 Choisissez une mammographie",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Formats supportés: JPG, JPEG, PNG, BMP, TIFF"
    )
    
    if uploaded is None:
        st.info("👆 Téléchargez une image pour commencer l'analyse")
        return
    
    # Chargement et affichage
    pil_img = Image.open(uploaded).convert("L")
    img_array = np.array(pil_img)
    
    # Paramètres simplifiés
    conf_threshold = 0.15
    
    # Affichage de l'image
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 📷 Image originale")
        st.image(img_array, use_column_width=True, channels="GRAY")
    
    with col_right:
        st.markdown("### 🔬 Analyse")
        
        if st.button("🚀 Lancer l'analyse", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                rapport = analyser(img_array, model, conf_threshold)
                img_annotee = annoter_image(img_array, rapport)
            
            # Affichage des résultats
            st.image(img_annotee, use_column_width=True)
            
           
            # Recommandation
            st.markdown(f"""
                <div class="info-box">
                    <strong>💡 Recommandation:</strong><br>
                    {rapport.recommandation}
                </div>
            """, unsafe_allow_html=True)
            
            # Détails des lésions
            if rapport.lesions:
                st.markdown("### 📋 Détail des lésions")
                
                # Tableau des lésions
                data = []
                for i, l in enumerate(rapport.lesions, 1):
                    data.append({
                        "N°": i,
                        "Type": l["type_lesion"],
                        "BI-RADS": l["birads"],
                        "Détails": str(l["details"])[:50] + "..."
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
               

if __name__ == "__main__":
    main()
