#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cascade_birads_final.py — Pipeline cascade complet
==============================================================
YOLO detecte les lesions (Mass/Calcification)
  -> pour chaque boite, analyse morphologique OpenCV
  -> regle clinique -> BI-RADS 2/3/4
  -> BI-RADS image = max des lesions

Usage :
    python cascade_birads_final.py image.jpg
    python cascade_birads_final.py image.jpg --json
    python cascade_birads_final.py image.jpg --no-viz

Prerequis :
    pip install ultralytics opencv-python numpy
    + le modele YOLO entraine (best.pt) -> voir MODELE_PATH
==============================================================
"""
import cv2
import numpy as np
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List
from ultralytics import YOLO

# ==========================================================
# CONFIGURATION
# ==========================================================
MODELE_PATH = "best.onnx"   # <-- ton best.pt YOLO renomme
CONF_SEUIL  = 0.10                          # seuil de detection (bas = moins de lesions ratees)
IMGSZ       = 1024
CLASSES_YOLO = {0: "Mass", 1: "Calcification"}
MARGE_CROP  = 0.15                          # elargir la boite de 15% pour le contexte

# ==========================================================
# STRUCTURES
# ==========================================================
@dataclass
class LesionAnalysee:
    type_lesion: str            # Mass / Calcification
    birads:      str            # BI-RADS 2/3/4
    confiance_yolo: float
    # champs masse
    forme:       Optional[str] = None
    contour:     Optional[str] = None
    densite:     Optional[str] = None
    score_suspicion: Optional[float] = None
    # champs calcification
    nb_calcif:   Optional[int] = None
    forme_calcif: Optional[str] = None
    repartition: Optional[str] = None
    benigne:     Optional[bool] = None
    box: Optional[tuple] = None

@dataclass
class RapportCascade:
    fichier:      str
    nb_lesions:   int
    lesions:      List[dict]
    birads_final: str
    recommandation: str

BIRADS_RECO = {
    "BI-RADS 1": "Mammographie normale — aucune lesion detectee.",
    "BI-RADS 2": "Anomalie benigne — surveillance standard.",
    "BI-RADS 3": "Anomalie probablement benigne — surveillance court terme (4-6 mois).",
    "BI-RADS 4": "Anomalie suspecte — verification histologique (biopsie) recommandee.",
}

# ==========================================================
# FEATURES MORPHOLOGIQUES (tes fonctions, conservees)
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
# ANALYSE d'une MASSE detectee par YOLO
# ==========================================================
def analyser_crop_masse(crop_gray) -> dict:
    """Segmente la masse dans le crop YOLO puis l'analyse."""
    blur = cv2.GaussianBlur(crop_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"forme":"indistinct","contour":"indistinct","densite":"egale","score":0.3}
    c = max(contours, key=cv2.contourArea)
    f = calculer_features_contour(c)
    forme   = classer_forme(f)
    contour = classer_contour_masse(crop_gray, c, f)
    densite, _ = classer_densite_masse(crop_gray, c)
    score   = calculer_score_suspicion(forme, contour, densite)
    return {"forme":forme,"contour":contour,"densite":densite,"score":round(score,2)}

def masse_vers_birads(score: float) -> str:
    """Regle : score de suspicion -> BI-RADS."""
    if score >= 0.45:
        return "BI-RADS 4"
    elif score >= 0.15:
        return "BI-RADS 3"
    return "BI-RADS 2"

# ==========================================================
# ANALYSE d'une CALCIFICATION detectee par YOLO
# ==========================================================
def analyser_crop_calcif(crop_gray) -> dict:
    """Compte et caracterise les calcifications dans le crop YOLO."""
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(crop_gray)
    se       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat   = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, se)
    seuil    = max(np.percentile(tophat, 97) if tophat.size else 30, 20)
    _, thr   = cv2.threshold(tophat, seuil, 255, cv2.THRESH_BINARY)
    k        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned  = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
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

    # Forme
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

    # Benignite (grossieres ou vasculaires)
    benigne = False
    if tailles and np.mean(tailles) > 8:
        benigne = True
    if elongations and np.mean(elongations) > 3.5:
        benigne = True

    return {"nb_calcif":nb, "forme_calcif":forme, "irregulieres":irregulieres,
            "benigne":benigne}

def calcif_vers_birads(info: dict) -> str:
    """Regle : caracteristiques des calcifications -> BI-RADS."""
    nb = info["nb_calcif"]
    if info["benigne"]:
        return "BI-RADS 2"
    if nb >= 10 and info["irregulieres"] >= 3:
        return "BI-RADS 4"          # nombreuses + polymorphes
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

def analyser_image(chemin_image: str, model) -> RapportCascade:
    img_gray = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Image introuvable : {chemin_image}")

    # 1. YOLO detecte les lesions
    results = model.predict(chemin_image, conf=CONF_SEUIL, iou=0.5,
                            imgsz=IMGSZ, verbose=False)[0]

    lesions = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            # elargir la boite (contexte)
            bw, bh = x2-x1, y2-y1
            mx, my = int(bw*MARGE_CROP), int(bh*MARGE_CROP)
            x1 = max(0, x1-mx); y1 = max(0, y1-my)
            x2 = min(img_gray.shape[1], x2+mx); y2 = min(img_gray.shape[0], y2+my)
            crop = img_gray[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            type_lesion = CLASSES_YOLO.get(cls_id, "Mass")
            if type_lesion == "Mass":
                a = analyser_crop_masse(crop)
                birads = masse_vers_birads(a["score"])
                lesions.append(LesionAnalysee(
                    type_lesion="Mass", birads=birads, confiance_yolo=round(conf,2),
                    forme=a["forme"], contour=a["contour"], densite=a["densite"],
                    score_suspicion=a["score"], box=(x1,y1,x2,y2)))
            else:
                a = analyser_crop_calcif(crop)
                birads = calcif_vers_birads(a)
                lesions.append(LesionAnalysee(
                    type_lesion="Calcification", birads=birads, confiance_yolo=round(conf,2),
                    nb_calcif=a["nb_calcif"], forme_calcif=a["forme_calcif"],
                    benigne=a["benigne"], box=(x1,y1,x2,y2)))

    # 2. BI-RADS image = max
    if lesions:
        niv_max = max(niveau_num(l.birads) for l in lesions)
        birads_final = f"BI-RADS {niv_max}"
    else:
        birads_final = "BI-RADS 1"

    return RapportCascade(
        fichier=Path(chemin_image).name,
        nb_lesions=len(lesions),
        lesions=[asdict(l) for l in lesions],
        birads_final=birads_final,
        recommandation=BIRADS_RECO.get(birads_final, ""),
    )

# ==========================================================
# VISUALISATION
# ==========================================================
def visualiser(chemin_image, rapport, model):
    img = cv2.imread(chemin_image)
    couleurs = {"BI-RADS 2":(0,180,0), "BI-RADS 3":(0,180,255), "BI-RADS 4":(0,0,220)}
    for l in rapport.lesions:
        if not l.get("box"): continue
        x1,y1,x2,y2 = l["box"]
        c = couleurs.get(l["birads"], (200,200,200))
        cv2.rectangle(img, (x1,y1), (x2,y2), c, 3)
        cv2.putText(img, f"{l['type_lesion'][:4]} {l['birads']}",
                    (x1, max(y1-8,20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    h = img.shape[0]
    cv2.rectangle(img, (0,h-40), (img.shape[1],h), (20,20,20), -1)
    cf = couleurs.get(rapport.birads_final,(150,150,150))
    cv2.putText(img, f"Conclusion: {rapport.birads_final}", (10,h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cf, 2)
    sortie = Path(chemin_image).stem + "_cascade.jpg"
    cv2.imwrite(sortie, img)
    return sortie

def afficher_rapport(r: RapportCascade):
    sep = "="*54
    print(f"\n{sep}\n  RAPPORT CASCADE — {r.fichier}\n{sep}")
    print(f"  Lesions detectees : {r.nb_lesions}")
    for i, l in enumerate(r.lesions, 1):
        print(f"\n  Lesion {i} : {l['type_lesion']} -> {l['birads']} (conf {l['confiance_yolo']})")
        if l['type_lesion']=="Mass":
            print(f"     forme={l['forme']} | contour={l['contour']} | densite={l['densite']} | score={l['score_suspicion']}")
        else:
            print(f"     nb_calcif={l['nb_calcif']} | forme={l['forme_calcif']} | benigne={l['benigne']}")
    print(f"\n  >>> CONCLUSION : {r.birads_final}")
    print(f"      {r.recommandation}\n{sep}\n")

# ==========================================================
# POINT D'ENTREE
# ==========================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python cascade_birads_final.py <image> [--json] [--no-viz]")
        sys.exit(1)
    chemin = sys.argv[1]
    if not Path(chemin).exists():
        print(f"Erreur : fichier introuvable -> {chemin}"); sys.exit(1)

    print("Chargement du modele YOLO...")
    try:
        model = YOLO(MODELE_PATH)
    except Exception as e:
        print(f"Erreur chargement modele ({MODELE_PATH}) : {e}"); sys.exit(1)

    print(f"Analyse de {chemin}...")
    rapport = analyser_image(chemin, model)

    if "--json" in sys.argv:
        print(json.dumps(asdict(rapport), indent=2, ensure_ascii=False))
    else:
        afficher_rapport(rapport)

    if "--no-viz" not in sys.argv:
        sortie = visualiser(chemin, rapport, model)
        print(f"Image annotee -> {sortie}")
