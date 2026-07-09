#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_cascade_streamlit.py — Pipeline HYBRIDE + interface Streamlit interactive
==============================================================================
- Rognage de l'image A LA SOURIS (streamlit-cropper)
- Curseurs pour ajuster TOUS les parametres en direct
- VOIE 1 : YOLO -> masses | VOIE 2 : Top-Hat -> microcalcifications
==============================================================================
LANCER :  streamlit run app_cascade_streamlit.py
PREREQUIS :
    pip install ultralytics opencv-python numpy streamlit pillow streamlit-cropper
"""
import cv2
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import tempfile, os

# rognage souris
try:
    from streamlit_cropper import st_cropper
    CROPPER_OK = True
except ImportError:
    CROPPER_OK = False

# ==========================================================
# CONFIG PAR DEFAUT (modifiables via les curseurs)
# ==========================================================
MODELE_PATH = "detecteur_masscalcif.pt"
IMGSZ       = 1024
MARGE_CROP  = 0.15

BIRADS_RECO = {
    "BI-RADS 1":"Mammographie normale — aucune lesion detectee.",
    "BI-RADS 2":"Anomalie benigne — surveillance standard.",
    "BI-RADS 3":"Anomalie probablement benigne — surveillance court terme.",
    "BI-RADS 4":"Anomalie suspecte — biopsie recommandee.",
}

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
    _, m = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    contours,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(img_gray)*255
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_gray); cv2.drawContours(mask,[c],-1,255,-1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
    return cv2.erode(mask, k2, iterations=2)

# ==========================================================
# MASSES (YOLO)
# ==========================================================
def features(contour):
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

def classer_contour_masse(g,contour,f):
    if not f: return "indistinct"
    r=f.get("rugosite",1); dc=f.get("deficit_convexite",0); cv_=f.get("convexite",1)
    if r>25 and dc>0.15: return "spicule"
    elif 10<r<=25 and cv_<0.85: return "microlobule"
    elif cv_>0.90: return "circumscrit"
    return "indistinct"

def classer_densite_masse(g,contour):
    mi=np.zeros(g.shape,np.uint8); cv2.drawContours(mi,[contour],-1,255,-1)
    me=np.zeros(g.shape,np.uint8); cv2.drawContours(me,[contour],-1,255,30)
    env=cv2.bitwise_xor(me,mi); pm=g[mi>0]; pe=g[env>0]
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

def masse_vers_birads(score, seuil_b3, seuil_b4):
    if score>=seuil_b4: return "BI-RADS 4"
    elif score>=seuil_b3: return "BI-RADS 3"
    return "BI-RADS 2"

def analyser_masse(crop):
    blur=cv2.GaussianBlur(crop,(5,5),0)
    _,th=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cont,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cont: return {"forme":"indistinct","contour":"indistinct","densite":"egale","score":0.3}
    c=max(cont,key=cv2.contourArea); f=features(c)
    forme=classer_forme(f); contour=classer_contour_masse(crop,c,f)
    densite=classer_densite_masse(crop,c); score=score_masse(forme,contour,densite)
    return {"forme":forme,"contour":contour,"densite":densite,"score":round(score,2)}

def detecter_masses_yolo(img_gray, model, conf, seuil_b3, seuil_b4):
    img_3c = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    res = model.predict(img_3c, conf=conf, iou=0.5, imgsz=IMGSZ, verbose=False)[0]
    masses=[]
    if res.boxes is not None:
        for box in res.boxes:
            if int(box.cls[0])!=0: continue
            cf=float(box.conf[0])
            x1,y1,x2,y2=[int(v) for v in box.xyxy[0]]
            bw,bh=x2-x1,y2-y1; mx,my=int(bw*MARGE_CROP),int(bh*MARGE_CROP)
            x1=max(0,x1-mx); y1=max(0,y1-my)
            x2=min(img_gray.shape[1],x2+mx); y2=min(img_gray.shape[0],y2+my)
            crop=img_gray[y1:y2,x1:x2]
            if crop.size==0: continue
            a=analyser_masse(crop)
            masses.append(Lesion("Mass", masse_vers_birads(a["score"],seuil_b3,seuil_b4),
                {"forme":a["forme"],"contour":a["contour"],"densite":a["densite"],
                 "score":a["score"],"conf_yolo":round(cf,2)}, (x1,y1,x2,y2)))
    return masses

# ==========================================================
# CALCIFICATIONS (Top-Hat) — parametrable
# ==========================================================
def detecter_calcifications(img_gray, se_size, aire_min, aire_max, percentile):
    mask = masque_sein(img_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enh = clahe.apply(img_gray)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    tophat = cv2.morphologyEx(enh, cv2.MORPH_TOPHAT, se)
    zone = tophat[mask>0]
    seuil = max(np.percentile(zone, percentile) if zone.size else 30, 15)
    _, thr = cv2.threshold(tophat, seuil, 255, cv2.THRESH_BINARY)
    thr = cv2.bitwise_and(thr, thr, mask=mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
    contours,_ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H,W=img_gray.shape
    points, tailles, elongs = [], [], []
    for c in contours:
        a=cv2.contourArea(c)
        if not (aire_min<=a<=aire_max): continue
        x,y,bw,bh=cv2.boundingRect(c)
        if x<30 or y<30 or (x+bw)>(W-30) or (y+bh)>(H-30): continue
        if len(c)>=5:
            try:
                ell=cv2.fitEllipse(c); elong=max(ell[1])/max(min(ell[1]),0.1)
                if elong>6.0: continue
                elongs.append(elong)
            except: pass
        M=cv2.moments(c)
        if M["m00"]>0:
            points.append((M["m10"]/M["m00"], M["m01"]/M["m00"])); tailles.append(np.sqrt(a))
    return points, tailles, elongs

def regrouper_en_amas(points, img_w, cluster_min, cluster_radius):
    if len(points) < cluster_min: return []
    pts=np.array(points); rayon=cluster_radius*img_w
    na=list(range(len(pts))); amas=[]
    while na:
        base=na.pop(0); groupe=[base]; i=0
        while i<len(na):
            idx=na[i]
            if any(np.linalg.norm(pts[idx]-pts[g])<rayon for g in groupe):
                groupe.append(idx); na.pop(i); i=0
            else: i+=1
        if len(groupe)>=cluster_min:
            amas.append([tuple(pts[g]) for g in groupe])
    return amas

def calcif_vers_birads(nb, benigne, polymorphe):
    if benigne: return "BI-RADS 2"
    if nb>=10 and polymorphe: return "BI-RADS 4"
    if nb>=10: return "BI-RADS 3"
    if nb<3: return "BI-RADS 2"
    return "BI-RADS 3"

def analyser_amas(amas_pts, tailles, elongs):
    nb=len(amas_pts)
    tm=np.mean(tailles) if tailles else 0
    em=np.mean(elongs) if elongs else 1.0
    benigne=(tm>8) or (em>3.5)
    poly=(np.std(tailles)>2.0) if len(tailles)>1 else False
    return calcif_vers_birads(nb,benigne,poly), {
        "nb_calcif":nb,"taille_moy":round(tm,1),"benigne":bool(benigne),"polymorphe":bool(poly)}

# ==========================================================
# CASCADE
# ==========================================================
def niveau(b): return int(b.split()[-1])

def analyser(img_gray, model, params):
    H,W = img_gray.shape
    masses = detecter_masses_yolo(img_gray, model, params["conf"],
                                  params["seuil_b3"], params["seuil_b4"])
    points, tailles, elongs = detecter_calcifications(
        img_gray, params["se_size"], params["aire_min"],
        params["aire_max"], params["percentile"])
    amas = regrouper_en_amas(points, W, params["cluster_min"], params["cluster_radius"])
    lesions=list(masses)
    for ap in amas:
        birads, details = analyser_amas(ap, tailles, elongs)
        xs=[p[0] for p in ap]; ys=[p[1] for p in ap]
        box=(int(min(xs))-10,int(min(ys))-10,int(max(xs))+10,int(max(ys))+10)
        lesions.append(Lesion("Calcification", birads, details, box))
    birads_final = f"BI-RADS {max(niveau(l.birads) for l in lesions)}" if lesions else "BI-RADS 1"
    return Rapport("image", len(masses), len(amas),
                   [asdict(l) for l in lesions], birads_final,
                   BIRADS_RECO.get(birads_final,""))

def dessiner(img_gray, rapport):
    img=cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    couleurs={"BI-RADS 2":(0,180,0),"BI-RADS 3":(0,180,255),"BI-RADS 4":(0,0,220)}
    for l in rapport.lesions:
        if not l.get("box"): continue
        x1,y1,x2,y2=l["box"]; c=couleurs.get(l["birads"],(200,200,200))
        cv2.rectangle(img,(x1,y1),(x2,y2),c,3)
        cv2.putText(img,f"{l['type_lesion'][:4]} {l['birads']}",(x1,max(y1-8,20)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ==========================================================
# INTERFACE STREAMLIT
# ==========================================================
def main():
    st.set_page_config(page_title="Analyse BI-RADS", layout="wide")
    st.title("🩺 Analyse Hybride Mammographie — Classification BI-RADS")

    @st.cache_resource
    def load_model():
        try: return YOLO(MODELE_PATH)
        except Exception as e:
            st.error(f"Erreur chargement modele : {e}"); return None
    model = load_model()
    if model is None: st.stop()

    # --- Curseurs de parametres (barre laterale) ---
    st.sidebar.header("⚙️ Parametres")
    st.sidebar.subheader("Masses (YOLO)")
    conf = st.sidebar.slider("Seuil confiance YOLO", 0.01, 0.50, 0.05, 0.01)
    seuil_b3 = st.sidebar.slider("Score -> BI-RADS 3", 0.05, 0.60, 0.15, 0.05)
    seuil_b4 = st.sidebar.slider("Score -> BI-RADS 4", 0.20, 0.90, 0.45, 0.05)
    st.sidebar.subheader("Calcifications (Top-Hat)")
    se_size = st.sidebar.slider("Taille element structurant", 11, 41, 25, 2)
    aire_min = st.sidebar.slider("Aire min (px2)", 1, 20, 2, 1)
    aire_max = st.sidebar.slider("Aire max (px2)", 30, 200, 90, 10)
    percentile = st.sidebar.slider("Percentile brillance", 95.0, 99.9, 99.0, 0.1)
    cluster_min = st.sidebar.slider("Min calcif / amas", 3, 20, 5, 1)
    cluster_radius = st.sidebar.slider("Rayon regroupement", 0.01, 0.15, 0.06, 0.01)

    params = {"conf":conf,"seuil_b3":seuil_b3,"seuil_b4":seuil_b4,
              "se_size":se_size,"aire_min":aire_min,"aire_max":aire_max,
              "percentile":percentile,"cluster_min":cluster_min,
              "cluster_radius":cluster_radius}

    uploaded = st.file_uploader("Choisissez une mammographie",
                                type=['jpg','jpeg','png','bmp','tiff'])
    if uploaded is None:
        st.info("Uploadez une image pour commencer.")
        return

    pil_img = Image.open(uploaded).convert("L")   # niveaux de gris

    # --- Rognage a la souris ---
    st.markdown("### ✂️ Rognez la zone du sein (glissez la souris)")
    if CROPPER_OK:
        cropped = st_cropper(pil_img, realtime_update=True, box_color="#00FF00",
                             aspect_ratio=None)
    else:
        st.warning("Module 'streamlit-cropper' non installe. "
                   "Installe-le avec : pip install streamlit-cropper "
                   "(l'image entiere est utilisee en attendant).")
        cropped = pil_img

    img_gray = np.array(cropped)

    if st.button("🔬 Analyser la zone rognee", type="primary"):
        with st.spinner("Analyse en cours..."):
            rapport = analyser(img_gray, model, params)
        col_img, col_res = st.columns([2,1])
        with col_img:
            st.markdown("### Image annotee")
            st.image(dessiner(img_gray, rapport), use_column_width=True)
        with col_res:
            st.markdown("## 📊 Resultats")
            c1,c2,c3 = st.columns(3)
            c1.metric("Masses", rapport.nb_masses)
            c2.metric("Amas calcif", rapport.nb_amas_calcif)
            c3.metric("BI-RADS", rapport.birads_final)
            st.info(rapport.recommandation)
            for i,l in enumerate(rapport.lesions,1):
                with st.expander(f"Lesion {i} : {l['type_lesion']} - {l['birads']}"):
                    st.json(l['details'])
            st.download_button("📥 Rapport JSON",
                json.dumps(asdict(rapport),indent=2,ensure_ascii=False),
                file_name="rapport.json", mime="application/json")

if __name__ == "__main__":
    main()