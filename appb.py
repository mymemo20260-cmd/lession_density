import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import json
import os

# ======================
# LOAD MODEL
# ======================
model = YOLO("best.onnx")

st.title("🧠 Détection de lésions mammaires")

# ======================
# CHARGER LES MÉTADONNÉES (BI-RADS)
# ======================
@st.cache_data
def load_birads():
    """Charge les BI-RADS depuis les métadonnées"""
    birads_dict = {}
    metadata_dir = "metadata"
    
    if os.path.exists(metadata_dir):
        for split in ["train", "val"]:
            split_dir = os.path.join(metadata_dir, split)
            if os.path.exists(split_dir):
                for file in os.listdir(split_dir):
                    if file.endswith('.json'):
                        try:
                            with open(os.path.join(split_dir, file), 'r') as f:
                                data = json.load(f)
                                # Associer l'ID de l'image au BI-RADS
                                # Ici on utilise le class_id pour simplifier
                                class_id = data.get('class_id', -1)
                                if class_id >= 0:
                                    birads_dict[class_id] = data.get('birads', 'BI-RADS 4')
                        except:
                            continue
    
    return birads_dict

birads_dict = load_birads()

def get_birads(class_id, class_name):
    """Retourne le BI-RADS pour une classe"""
    if class_id in birads_dict:
        return birads_dict[class_id]
    # Fallback
    if class_name == "Mass":
        return "BI-RADS 4"
    elif class_name == "Calcification":
        return "BI-RADS 4"
    return "BI-RADS 3"

# ======================
# UPLOAD IMAGE
# ======================
uploaded_file = st.file_uploader("Upload une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire l'image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Image originale", channels="BGR", use_column_width=True)

    # ======================
    # PREDICTION
    # ======================
    results = model.predict(source=img, conf=0.10)
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        names = results[0].names
        
        # Copie pour annotation
        annotated_img = img.copy()
        detections = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0].cpu().numpy())
            class_name = names[class_id] if class_id in names else "Inconnu"
            
            # Dessiner le rectangle
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Afficher le nom de la lésion
            cv2.putText(annotated_img, class_name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Récupérer le BI-RADS
            birads = get_birads(class_id, class_name)
            
            detections.append({
                "Lésion": class_name,
                "BI-RADS": birads
            })
        
        with col2:
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_img_rgb, caption="Résultat détection", use_column_width=True)
        
        # ======================
        # AFFICHAGE DU TABLEAU
        # ======================
        st.subheader("📋 Résultats")
        
        # Créer un DataFrame
        df = pd.DataFrame(detections)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Message de succès
        st.success(f"✅ {len(detections)} lésion(s) détectée(s)")
        
    else:
        st.warning("⚠️ Aucune lésion détectée")
        with col2:
            st.image(img, caption="Aucune détection", channels="BGR", use_column_width=True)
