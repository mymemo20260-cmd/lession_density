#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cascade_hybride_birads.py — Pipeline HYBRIDE
==============================================================
VOIE 1 : YOLO      -> detecte les MASSES (on ignore ses calcifs)
VOIE 2 : Top-Hat   -> detecte les MICROCALCIFICATIONS (morphologique)
  -> analyse de chaque lesion -> BI-RADS 2/3/4
  -> BI-RADS image = max

Usage :
    python cascade_hybride_birads.py image.jpg
    python cascade_hybride_birads.py image.jpg --json
    python cascade_hybride_birads.py image.jpg --no-viz

Prerequis : pip install ultralytics opencv-python numpy
            + best.pt YOLO -> MODELE_PATH
==============================================================
"""
import cv2
import numpy as np
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List
from ultralytics import YOLO

# ==========================================================
# CONFIGURATION
# ==========================================================
MODELE_PATH = "detecteur_masscalcif.pt"
CONF_MASSE  = 0.10          # seuil YOLO pour les masses
IMGSZ       = 1024
MARGE_CROP  = 0.15

# Parametres Top-Hat calcifications (a calibrer sur tes images)
TOPHAT_SE_SIZE   = 25       # taille element structurant
CALCIF_AIRE_MIN  = 2        # px^2
CALCIF_AIRE_MAX  = 90      # px^2
CALCIF_PERCENTILE = 99.0    # seuil de brillance (95-98)
CLUSTER_MIN      = 5        # min de calcifs pour un amas valide
CLUSTER_RADIUS   = 0.06     # rayon de regroupement (fraction de la largeur image)



# ==========================================================
# STRUCTURES
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

BIRADS_RECO = {
    "BI-RADS 1": "Mammographie normale — aucune lesion detectee.",
    "BI-RADS 2": "Anomalie benigne — surveillance standard.",
    "BI-RADS 3": "Anomalie probablement benigne — surveillance court terme.",
    "BI-RADS 4": "Anomalie suspecte — biopsie recommandee.",
}

# ==========================================================
# PRETRAITEMENT (masque du sein)
# ==========================================================
def masque_sein(img_gray):
    _, m = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    # Nettoyer le masque (retirer les petits artefacts des bords)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(img_gray) * 255
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_gray)
    cv2.drawContours(mask, [c], -1, 255, -1)
    # EROSION : on retire une bande sur le pourtour du sein (exclut les bords)
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    mask = cv2.erode(mask, k_erode, iterations=2)
    return mask

# ==========================================================
# VOIE 1 — MASSES via YOLO + tes fonctions d'analyse
# ==========================================================
def calculer_features_contour(contour):
    aire=cv2.contourArea(contour); perim=cv2.arcLength(contour,True)
    if perim==0 or aire==0: return {}
    circ=4*np.pi*aire/(perim**2)
    hull=cv2.convexHull(contour); ah=cv2.contourArea(hull)
    conv=aire/ah if ah>0 else 0; dc=(ah-aire)/ah if ah>0 else 0
    ra=1.0
    if len(contour)>=5:
        try:
            e=cv2.fitEllipse(contour); a,b=max(e[1]),min(e[1]); ra=a/b if b>0 else 1
        except: pass
    approx=cv2.approxPolyDP(contour,0.005*perim,True)
    return {"circularite":circ,"convexite":conv,"deficit_convexite":dc,
            "rapport_axes":ra,"rugosite":len(contour)/max(len(approx),1)}

def classer_forme(f):
    if not f: return "irreguliere"
    if f["circularite"]>=0.80 and f["rapport_axes"]<1.4: return "ronde_arrondie"
    elif f["circularite"]>=0.55 and f["rapport_axes"]<2.5 and f["convexite"]>0.80: return "ovale_elliptique"
    return "irreguliere"

def classer_contour_masse(g, contour, f):
    if not f: return "indistinct"
    r=f.get("rugosite",1); dc=f.get("deficit_convexite",0); cv_=f.get("convexite",1)
    if r>25 and dc>0.15: return "spicule"
    elif 10<r<=25 and cv_<0.85: return "microlobule"
    elif cv_>0.90: return "circumscrit"
    return "indistinct"

def classer_densite_masse(g, contour):
    mi=np.zeros(g.shape,np.uint8); cv2.drawContours(mi,[contour],-1,255,-1)
    me=np.zeros(g.shape,np.uint8); cv2.drawContours(me,[contour],-1,255,30)
    env=cv2.bitwise_xor(me,mi)
    pm=g[mi>0]; pe=g[env>0]
    if len(pm)==0 or len(pe)==0: return "egale"
    ratio=np.mean(pm)/max(np.mean(pe),1)
    if ratio>1.30: return "haute"
    elif ratio>1.05: return "egale"
    elif ratio>0.80: return "faible"
    return "graisseuse"

def score_masse(forme,contour,densite):
    s=0.0
    s+={"irreguliere":0.40,"ovale_elliptique":0.10,"ronde_arrondie":0.0}.get(forme,0.2)
    s+={"spicule":0.40,"microlobule":0.20,"indistinct":0.15,"circumscrit":0.0}.get(contour,0.15)
    s+={"haute":0.20,"egale":0.10,"faible":0.05,"graisseuse":0.0}.get(densite,0.1)
    return min(s,1.0)

def masse_vers_birads(score):
    if score>=0.45: return "BI-RADS 4"
    elif score>=0.15: return "BI-RADS 3"
    return "BI-RADS 2"

def analyser_masse(crop):
    blur=cv2.GaussianBlur(crop,(5,5),0)
    _,th=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cont,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cont: return {"forme":"indistinct","contour":"indistinct","densite":"egale","score":0.3}
    c=max(cont,key=cv2.contourArea); f=calculer_features_contour(c)
    forme=classer_forme(f); contour=classer_contour_masse(crop,c,f)
    densite=classer_densite_masse(crop,c); score=score_masse(forme,contour,densite)
    return {"forme":forme,"contour":contour,"densite":densite,"score":round(score,2)}

def detecter_masses_yolo(img_gray, model):
    # YOLO attend 3 canaux : on convertit le gris en BGR (3 canaux identiques)
    img_3c = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    res = model.predict(img_3c, conf=CONF_MASSE, iou=0.5, imgsz=IMGSZ, verbose=False)[0]
    masses = []
    if res.boxes is not None:
        for box in res.boxes:
            if int(box.cls[0]) != 0:   # 0 = Mass ; on IGNORE les calcifs de YOLO
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            bw, bh = x2 - x1, y2 - y1
            mx, my = int(bw * MARGE_CROP), int(bh * MARGE_CROP)
            x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
            x2 = min(img_gray.shape[1], x2 + mx); y2 = min(img_gray.shape[0], y2 + my)
            crop = img_gray[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            a = analyser_masse(crop)
            birads = masse_vers_birads(a["score"])
            masses.append(Lesion("Mass", birads,
                {"forme": a["forme"], "contour": a["contour"], "densite": a["densite"],
                 "score": a["score"], "conf_yolo": round(conf, 2)}, (x1, y1, x2, y2)))
    return masses

# ==========================================================
# VOIE 2 — MICROCALCIFICATIONS via Top-Hat
# ==========================================================
def detecter_calcifications_tophat(img_gray):
    """Detecte les microcalcifications par Top-Hat, puis les regroupe en amas."""
    mask = masque_sein(img_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh = clahe.apply(img_gray)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_SE_SIZE, TOPHAT_SE_SIZE))
    tophat = cv2.morphologyEx(enh, cv2.MORPH_TOPHAT, se)

    zone = tophat[mask>0]
    seuil = max(np.percentile(zone, CALCIF_PERCENTILE) if zone.size else 30, 15)
    _, thr = cv2.threshold(tophat, seuil, 255, cv2.THRESH_BINARY)
    thr = cv2.bitwise_and(thr, thr, mask=mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
    contours,_ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Points = centres des calcifications valides
    points, tailles, elongations = [], [], []
    for c in contours:
        a = cv2.contourArea(c)
        if not (CALCIF_AIRE_MIN <= a <= CALCIF_AIRE_MAX):
            continue
        # rejeter les structures allongees (vaisseaux)
        if len(c)>=5:
            try:
                ell=cv2.fitEllipse(c); elong=max(ell[1])/max(min(ell[1]),0.1)
                if elong>6.0: continue   # plus tolerant : garde les calcif en batonnets
                elongations.append(elong)
            except: pass
        M=cv2.moments(c)
        if M["m00"]>0:
            points.append((M["m10"]/M["m00"], M["m01"]/M["m00"]))
            tailles.append(np.sqrt(a))
    return points, tailles, elongations

def regrouper_en_amas(points, img_w):
    """Regroupe les points proches en amas (clustering simple par distance)."""
    if len(points) < CLUSTER_MIN:
        return []
    pts = np.array(points)
    rayon = CLUSTER_RADIUS * img_w
    non_assignes = list(range(len(pts)))
    amas = []
    while non_assignes:
        base = non_assignes.pop(0)
        groupe = [base]
        i = 0
        while i < len(non_assignes):
            idx = non_assignes[i]
            # proche d'au moins un point du groupe ?
            if any(np.linalg.norm(pts[idx]-pts[g]) < rayon for g in groupe):
                groupe.append(idx); non_assignes.pop(i); i=0
            else:
                i+=1
        if len(groupe) >= CLUSTER_MIN:
            amas.append([tuple(pts[g]) for g in groupe])
    return amas

def calcif_vers_birads(nb, benigne, polymorphe):
    if benigne: return "BI-RADS 2"
    if nb >= 10 and polymorphe: return "BI-RADS 4"
    if nb >= 10: return "BI-RADS 3"
    if nb < 3: return "BI-RADS 2"
    return "BI-RADS 3"

def analyser_amas_calcif(amas_points, tailles_all, elong_all):
    nb = len(amas_points)
    taille_moy = np.mean(tailles_all) if tailles_all else 0
    elong_moy  = np.mean(elong_all) if elong_all else 1.0
    benigne = (taille_moy > 8) or (elong_moy > 3.5)  # grossieres ou vasculaires
    # polymorphe : forte variance de taille
    polymorphe = (np.std(tailles_all) > 2.0) if len(tailles_all) > 1 else False
    birads = calcif_vers_birads(nb, benigne, polymorphe)
    return birads, {"nb_calcif":nb, "taille_moy":round(taille_moy,1),
                    "benigne":benigne, "polymorphe":bool(polymorphe)}

# ==========================================================
# CASCADE HYBRIDE
# ==========================================================
def niveau(b): return int(b.split()[-1])

def analyser_image(chemin, model):
    img = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(chemin)
    # Securite : si l'image a plusieurs canaux, la convertir en niveaux de gris
# Securite : normaliser en 2D niveaux de gris quel que soit le format
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]          # (H, W, 1) -> (H, W)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # (H, W, 3) -> (H, W)
    H, W = img.shape[:2]

    # VOIE 1 : masses
    masses = detecter_masses_yolo(img, model)

    # VOIE 2 : calcifications
    points, tailles, elongs = detecter_calcifications_tophat(img)
    amas = regrouper_en_amas(points, W)

    lesions = list(masses)
    for amas_pts in amas:
        # tailles/elongations de cet amas (approx : on prend l'ensemble)
        birads, details = analyser_amas_calcif(amas_pts, tailles, elongs)
        xs = [p[0] for p in amas_pts]; ys=[p[1] for p in amas_pts]
        box = (int(min(xs))-10, int(min(ys))-10, int(max(xs))+10, int(max(ys))+10)
        lesions.append(Lesion("Calcification", birads, details, box))

    # BI-RADS final = max
    if lesions:
        birads_final = f"BI-RADS {max(niveau(l.birads) for l in lesions)}"
    else:
        birads_final = "BI-RADS 1"

    return Rapport(
        fichier=Path(chemin).name,
        nb_masses=len(masses),
        nb_amas_calcif=len(amas),
        lesions=[asdict(l) for l in lesions],
        birads_final=birads_final,
        recommandation=BIRADS_RECO.get(birads_final,""),
    )

# ==========================================================
# VISUALISATION + AFFICHAGE
# ==========================================================
def visualiser(chemin, rapport):
    img = cv2.imread(chemin)
    couleurs = {"BI-RADS 2":(0,180,0),"BI-RADS 3":(0,180,255),"BI-RADS 4":(0,0,220)}
    for l in rapport.lesions:
        if not l.get("box"): continue
        x1,y1,x2,y2 = l["box"]; c=couleurs.get(l["birads"],(200,200,200))
        cv2.rectangle(img,(x1,y1),(x2,y2),c,3)
        cv2.putText(img,f"{l['type_lesion'][:4]} {l['birads']}",(x1,max(y1-8,20)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)
    h=img.shape[0]
    cv2.rectangle(img,(0,h-40),(img.shape[1],h),(20,20,20),-1)
    cf=couleurs.get(rapport.birads_final,(150,150,150))
    cv2.putText(img,f"Conclusion: {rapport.birads_final}",(10,h-14),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,cf,2)
    sortie=Path(chemin).stem+"_hybride.jpg"; cv2.imwrite(sortie,img)
    return sortie

def afficher(r):
    sep="="*54
    print(f"\n{sep}\n  RAPPORT HYBRIDE — {r.fichier}\n{sep}")
    print(f"  Masses (YOLO)          : {r.nb_masses}")
    print(f"  Amas calcif (Top-Hat)  : {r.nb_amas_calcif}")
    for i,l in enumerate(r.lesions,1):
        print(f"\n  Lesion {i} : {l['type_lesion']} -> {l['birads']}")
        for k,v in l['details'].items():
            print(f"     {k} = {v}")
    print(f"\n  >>> CONCLUSION : {r.birads_final}")
    print(f"      {r.recommandation}\n{sep}\n")

# ==========================================================
if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage : python cascade_hybride_birads.py <image> [--json] [--no-viz]")
        sys.exit(1)
    chemin=sys.argv[1]
    if not Path(chemin).exists():
        print(f"Fichier introuvable : {chemin}"); sys.exit(1)
    print("Chargement YOLO...")
    try:
        model=YOLO(MODELE_PATH)
    except Exception as e:
        print(f"Erreur modele : {e}"); sys.exit(1)
    print(f"Analyse de {chemin}...")
    rapport=analyser_image(chemin, model)
    if "--json" in sys.argv:
        print(json.dumps(asdict(rapport),indent=2,ensure_ascii=False))
    else:
        afficher(rapport)
    if "--no-viz" not in sys.argv:
        print("Image annotee ->", visualiser(chemin, rapport))
