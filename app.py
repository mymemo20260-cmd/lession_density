#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cascade_birads_streamlit.py — Pipeline cascade complet avec interface Streamlit
================================================================================
YOLO detecte les lesions (Mass/Calcification)
  -> Pre-traitement de l'image pour les mammographies
  -> pour chaque boite, analyse morphologique OpenCV
  -> regle clinique -> BI-RADS 2/3/4
  -> BI-RADS image = max des lesions

Utilisation :
    streamlit run cascade_birads_streamlit.py
================================================================================
"""
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List
from ultralytics import YOLO

# ==========================================================
# CONFIGURATION
# ==========================================================
MODELE_PATH = "best.onnx"
CONF_SEUIL  = 0.05
IMGSZ       = 640
CLASSES_YOLO = {0: "Mass", 1: "Calcification"}
MARGE_CROP  = 0.15

# ==========================================================
# PRETRAITEMENT POUR MAMMOGRAPHIES
# ==========================================================
def preprocess_mammogram(img):
    """
    Prétraitement spécifique pour les mammographies
    Retourne une image en couleur (3 canaux) pour YOLO
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 1. Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Débruitage
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # 3. Égalisation d'histogramme adaptative
    equalized = cv2.equalizeHist(denoised)
    
    # 4. Renforcement des bords
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(equalized, -1, kernel)
    
    # 5. Normalisation
    normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
    
    # 6. Filtre médian
    final_gray = cv2.medianBlur(normalized, 3)
    
    # 7. Convertir en 3 canaux pour YOLO
    final_bgr = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)
    
    # Sauvegarder les étapes pour visualisation
    steps = {
        "original": img,
        "gray": gray,
        "enhanced": enhanced,
        "denoised": denoised,
        "equalized": equalized,
        "sharpened": sharpened,
        "normalized": normalized,
        "final_gray": final_gray,
        "final_color": final_bgr
    }
    
    return final_bgr, steps

def preprocess_detection_region(roi):
    """
    Prétraitement spécifique pour la région d'intérêt (ROI)
    """
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

def preprocess_calcification_region(roi):
    """
    Prétraitement spécifique pour les calcifications
    """
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
    denoised = cv2.fastNlMeansDenoising(tophat, None, 10, 7, 21)
    
    return denoised

# ==========================================================
# STRUCTURES
# ==========================================================
@dataclass
class LesionAnalysee:
    type_lesion: str
    birads:      str
    forme:       Optional[str] = None
    contour:     Optional[str] = None
    densite:     Optional[str] = None
    nb_calcif:   Optional[int] = None
    forme_calcif: Optional[str] = None
    benigne:     Optional[bool] = None
    box: Optional[tuple] = None
    crop_image: Optional[np.ndarray] = None

@dataclass
class RapportCascade:
    fichier:      str
    nb_lesions:   int
    lesions:      List[dict]
    birads_final: str
    recommandation: str
    preprocessing: Optional[dict] = None

BIRADS_RECO = {
    "BI-RADS 1": "Mammographie normale — aucune lesion detectee.",
    "BI-RADS 2": "Anomalie benigne — surveillance standard.",
    "BI-RADS 3": "Anomalie probablement benigne — surveillance court terme (4-6 mois).",
    "BI-RADS 4": "Anomalie suspecte — verification histologique (biopsie) recommandee.",
}

# ==========================================================
# FEATURES MORPHOLOGIQUES
# ==========================================================
def calculer_features_contour(contour) -> dict:
    aire      = cv2.contourArea(contour)
    perimetre = cv2.arcLength(contour, True)
    if perimetre == 0 or aire == 0:
        return {}
    circularite = 4 * np.pi * aire / (perimetre ** 2)
    hull      = cv2.convexHull(contour)
    aire_hull = cv2.contourArea(hull)
    convexite = aire / aire_hull if aire_hull > 0 else 0
    deficit_convexite = (aire_hull - aire) / aire_hull if aire_hull > 0 else 0
    rapport_axes = 1.0
    excentricite = 0.0
    if len(contour) >= 5:
        try:
            ellipse      = cv2.fitEllipse(contour)
            a, b         = max(ellipse[1]), min(ellipse[1])
            rapport_axes = a / b if b > 0 else 1
            excentricite = np.sqrt(1 - (b / a) ** 2) if a > 0 else 0
        except Exception:
            pass
    epsilon  = 0.005 * perimetre
    approx   = cv2.approxPolyDP(contour, epsilon, True)
    rugosite = len(contour) / max(len(approx), 1)
    x, y, w, h = cv2.boundingRect(contour)
    solidite   = aire / (w * h) if (w * h) > 0 else 0
    perim_hull       = cv2.arcLength(hull, True)
    ratio_perim_hull = perim_hull / perimetre if perimetre > 0 else 1
    return {
        "aire": aire, "perimetre": perimetre, "circularite": circularite,
        "convexite": convexite, "deficit_convexite": deficit_convexite,
        "rapport_axes": rapport_axes, "excentricite": excentricite,
        "rugosite": rugosite, "solidite": solidite,
        "ratio_perim_hull": ratio_perim_hull,
    }

def classer_forme(f: dict) -> str:
    if not f:
        return "irreguliere"
    if f["circularite"] >= 0.80 and f["rapport_axes"] < 1.4:
        return "ronde_arrondie"
    elif f["circularite"] >= 0.55 and f["rapport_axes"] < 2.5 and f["convexite"] > 0.80:
        return "ovale_elliptique"
    return "irreguliere"

def classer_contour_masse(img_gray, contour, f: dict) -> str:
    if not f:
        return "indistinct"
    mask_bord = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask_bord, [contour], -1, 255, 8)
    sobel_x  = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y  = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_bord = gradient[mask_bord > 0]
    grad_moy  = np.mean(grad_bord) if len(grad_bord) > 0 else 0
    mask_int = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask_int, [contour], -1, 255, -1)
    grad_int = gradient[mask_int > 0]
    grad_int_moy = np.mean(grad_int) if len(grad_int) > 0 else 1
    ratio_grad = grad_moy / max(grad_int_moy, 1)
    rugosite          = f.get("rugosite", 1)
    deficit_convexite = f.get("deficit_convexite", 0)
    convexite         = f.get("convexite", 1)
    if rugosite > 25 and deficit_convexite > 0.15:
        return "spicule"
    elif 10 < rugosite <= 25 and convexite < 0.85:
        return "microlobule"
    elif ratio_grad > 1.8 and convexite > 0.90:
        return "circumscrit"
    return "indistinct"

def classer_densite_masse(img_gray, contour) -> tuple:
    mask_int = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask_int, [contour], -1, 255, -1)
    mask_ext = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask_ext, [contour], -1, 255, 30)
    mask_env = cv2.bitwise_xor(mask_ext, mask_int)
    pixels_masse = img_gray[mask_int > 0]
    pixels_env   = img_gray[mask_env > 0]
    if len(pixels_masse) == 0 or len(pixels_env) == 0:
        return "egale", 1.0
    ratio = np.mean(pixels_masse) / max(np.mean(pixels_env), 1)
    if ratio > 1.30:   return "haute", ratio
    elif ratio > 1.05: return "egale", ratio
    elif ratio > 0.80: return "faible", ratio
    return "graisseuse", ratio

def calculer_score_suspicion(forme, contour, densite) -> float:
    score = 0.0
    score += {"irreguliere":0.40,"ovale_elliptique":0.10,"ronde_arrondie":0.0}.get(forme,0.2)
    score += {"spicule":0.40,"microlobule":0.20,"indistinct":0.15,"circumscrit":0.0}.get(contour,0.15)
    score += {"haute":0.20,"egale":0.10,"faible":0.05,"graisseuse":0.0}.get(densite,0.1)
    return min(score, 1.0)

# ==========================================================
# ANALYSE d'une MASSE
# ==========================================================
def analyser_crop_masse(crop_gray) -> dict:
    preprocessed = preprocess_detection_region(crop_gray)
    
    blur = cv2.GaussianBlur(preprocessed, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"forme":"indistinct","contour":"indistinct","densite":"egale"}
    
    c = max(contours, key=cv2.contourArea)
    f = calculer_features_contour(c)
    forme   = classer_forme(f)
    contour = classer_contour_masse(preprocessed, c, f)
    densite, _ = classer_densite_masse(preprocessed, c)
    score   = calculer_score_suspicion(forme, contour, densite)
    
    return {"forme":forme,"contour":contour,"densite":densite,"score":round(score,2)}

def masse_vers_birads(score: float) -> str:
    if score >= 0.45:
        return "BI-RADS 4"
    elif score >= 0.15:
        return "BI-RADS 3"
    return "BI-RADS 2"

# ==========================================================
# ANALYSE d'une CALCIFICATION
# ==========================================================
def analyser_crop_calcif(crop_gray) -> dict:
    preprocessed = preprocess_calcification_region(crop_gray)
    
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(preprocessed, cv2.MORPH_TOPHAT, se)
    seuil = max(np.percentile(tophat, 97) if tophat.size else 30, 20)
    _, thr = cv2.threshold(tophat, seuil, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tailles, circularites, elongations, irregulieres = [], [], [], 0
    for c in contours:
        a = cv2.contourArea(c)
        if not (1 <= a <= 150):
            continue
        p = cv2.arcLength(c, True)
        tailles.append(np.sqrt(a))
        if p > 0:
            circularites.append(4*np.pi*a/(p**2))
        if len(c) >= 5:
            try:
                ell = cv2.fitEllipse(c)
                elong = max(ell[1]) / max(min(ell[1]), 0.1)
                elongations.append(elong)
                if elong > 2.5:
                    irregulieres += 1
            except Exception:
                pass
    nb = len(tailles)

    if nb == 0:
        forme = "indeterminee"
    elif np.mean(tailles) > 8:
        forme = "grossieres"
    elif circularites and np.mean(circularites) > 0.72 and np.std(circularites) < 0.15:
        forme = "punctiformes_regulieres"
    elif circularites and np.mean(circularites) > 0.45:
        forme = "punctiformes_irregulieres"
    else:
        forme = "poudreuses"

    benigne = False
    if tailles and np.mean(tailles) > 8:
        benigne = True
    if elongations and np.mean(elongations) > 3.5:
        benigne = True

    return {"nb_calcif":nb, "forme_calcif":forme, "irregulieres":irregulieres,
            "benigne":benigne}

def calcif_vers_birads(info: dict) -> str:
    nb = info["nb_calcif"]
    if info["benigne"]:
        return "BI-RADS 2"
    if nb >= 10 and info["irregulieres"] >= 3:
        return "BI-RADS 4"
    if nb >= 10:
        return "BI-RADS 3"
    if nb < 3:
        return "BI-RADS 2"
    return "BI-RADS 3"

# ==========================================================
# CASCADE PRINCIPALE
# ==========================================================
def niveau_num(birads_str: str) -> int:
    return int(birads_str.split()[-1])

def analyser_image(img, model, show_preprocessing=False) -> RapportCascade:
    if isinstance(img, np.ndarray):
        img_original = img.copy()
        chemin = "image_upload"
    else:
        img_original = cv2.imread(img)
        if img_original is None:
            raise FileNotFoundError(f"Image introuvable : {img}")
        chemin = Path(img).name

    # Prétraitement
    preprocessed_img, preprocess_steps = preprocess_mammogram(img_original)
    
    # Détection YOLO
    results = model.predict(preprocessed_img, conf=CONF_SEUIL, iou=0.5,
                            imgsz=IMGSZ, verbose=False)[0]

    lesions = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            
            box_original = (x1, y1, x2, y2)
            
            # Crop de l'image originale pour l'affichage
            crop_img = img_original[y1:y2, x1:x2].copy()
            
            # Élargir pour l'analyse
            bw, bh = x2-x1, y2-y1
            mx, my = int(bw*MARGE_CROP), int(bh*MARGE_CROP)
            x1_crop = max(0, x1-mx)
            y1_crop = max(0, y1-my)
            x2_crop = min(img_original.shape[1], x2+mx)
            y2_crop = min(img_original.shape[0], y2+my)
            
            if len(img_original.shape) == 3:
                img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_original
                
            crop = img_gray[y1_crop:y2_crop, x1_crop:x2_crop]
            if crop.size == 0:
                continue

            type_lesion = CLASSES_YOLO.get(cls_id, "Mass")
            if type_lesion == "Mass":
                a = analyser_crop_masse(crop)
                birads = masse_vers_birads(a["score"])
                lesions.append(LesionAnalysee(
                    type_lesion="Mass", birads=birads,
                    forme=a["forme"], contour=a["contour"], densite=a["densite"],
                    box=box_original, crop_image=crop_img))
            else:
                a = analyser_crop_calcif(crop)
                birads = calcif_vers_birads(a)
                lesions.append(LesionAnalysee(
                    type_lesion="Calcification", birads=birads,
                    nb_calcif=a["nb_calcif"], forme_calcif=a["forme_calcif"],
                    benigne=a["benigne"], box=box_original, crop_image=crop_img))

    if lesions:
        niv_max = max(niveau_num(l.birads) for l in lesions)
        birads_final = f"BI-RADS {niv_max}"
    else:
        birads_final = "BI-RADS 1"

    preprocess_info = None
    if show_preprocessing:
        preprocess_info = {
            "steps": {
                "original": preprocess_steps["original"],
                "gray": preprocess_steps["gray"],
                "enhanced": preprocess_steps["enhanced"],
                "denoised": preprocess_steps["denoised"],
                "equalized": preprocess_steps["equalized"],
                "sharpened": preprocess_steps["sharpened"],
                "normalized": preprocess_steps["normalized"],
                "final_gray": preprocess_steps["final_gray"],
                "final_color": preprocess_steps["final_color"]
            }
        }

    return RapportCascade(
        fichier=chemin,
        nb_lesions=len(lesions),
        lesions=[asdict(l) for l in lesions],
        birads_final=birads_final,
        recommandation=BIRADS_RECO.get(birads_final, ""),
        preprocessing=preprocess_info
    )

# ==========================================================
# VISUALISATION
# ==========================================================
def visualiser_globale(img, rapport):
    img_annotated = img.copy()
    couleurs = {"BI-RADS 2":(0,180,0), "BI-RADS 3":(0,180,255), "BI-RADS 4":(0,0,220)}
    
    for l in rapport.lesions:
        if not l.get("box"): continue
        x1,y1,x2,y2 = l["box"]
        c = couleurs.get(l["birads"], (200,200,200))
        
        cv2.rectangle(img_annotated, (x1,y1), (x2,y2), c, 3)
        
        label = f"{l['type_lesion']} {l['birads']}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_annotated, (x1, max(y1-30, 0)), (x1+w+10, max(y1-5, 0)), c, -1)
        cv2.putText(img_annotated, label, (x1+5, max(y1-10, 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    h = img_annotated.shape[0]
    cv2.rectangle(img_annotated, (0,h-40), (img_annotated.shape[1],h), (20,20,20), -1)
    cf = couleurs.get(rapport.birads_final,(150,150,150))
    cv2.putText(img_annotated, f"Conclusion: {rapport.birads_final}", (10,h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cf, 2)
    
    return img_annotated

# ==========================================================
# AFFICHAGE D'UNE LÉSION INDIVIDUELLE
# ==========================================================
def afficher_lesion_individuelle(l, index):
    couleurs = {
        "BI-RADS 2": ("#00b300", "🟢"),
        "BI-RADS 3": ("#ffb300", "🟡"),
        "BI-RADS 4": ("#dc3545", "🔴")
    }
    color, emoji = couleurs.get(l['birads'], ("#808080", "⚪"))
    
    with st.container():
        st.markdown(f"""
        <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: {color}10;">
            <h3 style="color: {color}; margin: 0;">
                {emoji} Lésion #{index} - {l['type_lesion']}
            </h3>
            <h4 style="color: {color}; margin: 5px 0;">{l['birads']}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            if l.get('crop_image') is not None:
                crop = l['crop_image']
                h, w = crop.shape[:2]
                if len(crop.shape) == 2:
                    crop_color = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                else:
                    crop_color = crop.copy()
                
                cv2.rectangle(crop_color, (2, 2), (w-2, h-2), (0, 255, 0), 2)
                cv2.putText(crop_color, l['type_lesion'], (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                st.image(cv2.cvtColor(crop_color, cv2.COLOR_BGR2RGB), 
                        caption=f"Région analysée", use_container_width=True)
        
        with col2:
            if l['type_lesion'] == "Mass":
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
                    <p><b>📐 Forme:</b> {l.get('forme', '-')}</p>
                    <p><b>🔍 Contour:</b> {l.get('contour', '-')}</p>
                    <p><b>⚪ Densité:</b> {l.get('densite', '-')}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
                    <p><b>🔢 Nombre:</b> {l.get('nb_calcif', 0)}</p>
                    <p><b>📐 Forme:</b> {l.get('forme_calcif', '-')}</p>
                    <p><b>✅ Bénigne:</b> {'Oui' if l.get('benigne') else 'Non'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            if l.get('box'):
                x1,y1,x2,y2 = l['box']
                st.caption(f"📍 Position: [{x1}, {y1}] → [{x2}, {y2}]")

# ==========================================================
# AFFICHAGE DU PRETRAITEMENT
# ==========================================================
def afficher_preprocessing(preprocess_info):
    if not preprocess_info:
        return
    
    st.subheader("🔬 Étapes du prétraitement (mammographie)")
    
    steps = preprocess_info["steps"]
    
    if "final_color" in steps:
        st.image(cv2.cvtColor(steps["final_color"], cv2.COLOR_BGR2RGB), 
                caption="Image prétraitée (envoyée au modèle YOLO)", use_container_width=True)
    
    cols = st.columns(4)
    
    step_names = {
        "original": "Original",
        "gray": "Niveaux de gris",
        "enhanced": "CLAHE",
        "denoised": "Débruitage",
        "equalized": "Égalisation",
        "sharpened": "Renforcement",
        "normalized": "Normalisation",
        "final_gray": "Final (gris)"
    }
    
    for i, (key, name) in enumerate(step_names.items()):
        if key in steps:
            col_idx = i % 4
            with cols[col_idx]:
                img_to_show = steps[key]
                if len(img_to_show.shape) == 2:
                    img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_GRAY2RGB)
                st.image(img_to_show, caption=name, use_container_width=True)

# ==========================================================
# INTERFACE STREAMLIT
# ==========================================================
def main():
    st.set_page_config(page_title="Cascade BI-RADS", layout="wide")
    st.title("🧠 Détection de lésions mammaires - Pipeline Cascade")
    st.markdown("---")
    
    with st.expander("ℹ️ Information sur le pipeline"):
        st.markdown("""
        Ce pipeline utilise une approche en cascade avec prétraitement spécifique pour mammographies :
        
        1. **Prétraitement mammo** : CLAHE, débruitage, égalisation, renforcement des bords
        2. **YOLOv8** détecte les lésions (Masses et Calcifications)
        3. **Analyse morphologique** avec OpenCV pour chaque lésion
        4. **Règles cliniques** pour déterminer le BI-RADS
        5. **BI-RADS final** = le plus élevé des lésions détectées
        """)
    
    @st.cache_resource
    def load_model():
        try:
            return YOLO(MODELE_PATH)
        except Exception as e:
            st.error(f"Erreur chargement du modèle : {e}")
            return None
    
    model = load_model()
    if model is None:
        st.stop()
    
    uploaded_file = st.file_uploader("📤 Uploader une mammographie", 
                                     type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        show_preproc = st.checkbox("🔬 Afficher les étapes du prétraitement", value=False)
        
        with st.spinner("🔬 Analyse en cours..."):
            rapport = analyser_image(img, model, show_preprocessing=show_preproc)
        
        if show_preproc and rapport.preprocessing:
            afficher_preprocessing(rapport.preprocessing)
            st.markdown("---")
        
        st.subheader("📊 Vue d'ensemble")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                    caption="Image originale", use_container_width=True)
        
        with col2:
            img_annotated = visualiser_globale(img, rapport)
            st.image(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB), 
                    caption=f"Résultat: {rapport.birads_final}", use_container_width=True)
        
        st.markdown("---")
        
        if rapport.nb_lesions > 0:
            st.subheader(f"🔍 Détail des {rapport.nb_lesions} lésion(s) détectée(s)")
            
            birads_color = {
                "BI-RADS 1": "#28a745",
                "BI-RADS 2": "#28a745",
                "BI-RADS 3": "#ffc107",
                "BI-RADS 4": "#dc3545"
            }.get(rapport.birads_final, "#808080")
            
            st.markdown(f"""
            <div style="padding: 15px; background-color: {birads_color}20; 
                        border-radius: 10px; border: 2px solid {birads_color}; margin-bottom: 20px;">
                <h3 style="color: {birads_color}; margin: 0;">
                    🎯 Conclusion: {rapport.birads_final}
                </h3>
                <p style="margin-top: 5px;">{rapport.recommandation}</p>
            </div>
            """, unsafe_allow_html=True)
            
            for i, l in enumerate(rapport.lesions, 1):
                afficher_lesion_individuelle(l, i)
                st.markdown("---")
            
            if st.button("📥 Télécharger le rapport complet (JSON)"):
                rapport_json = json.dumps(asdict(rapport), indent=2, ensure_ascii=False)
                st.download_button(
                    label="Télécharger",
                    data=rapport_json,
                    file_name=f"{Path(uploaded_file.name).stem}_rapport.json",
                    mime="application/json"
                )
        
        else:
            st.info("✅ Aucune lésion détectée - BI-RADS 1")
            st.balloons()

if __name__ == "__main__":
    main()
