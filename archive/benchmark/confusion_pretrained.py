import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# PARAMÈTRES DU TEST
# ==========================================
dossier_images = "images/test"
dossier_labels = "labels/test"
seuil_iou = 0.4
seuil_confiance = 0.10

def calculer_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea) if float(boxAArea + boxBArea - interArea) > 0 else 0

def dessiner_matrice(TP, FP, FN, nom_modele):
    """Génère une image PNG de la matrice de confusion (Style Seaborn Blues)"""
    # En détection d'objet, le "Vrai Négatif" (TN) du fond de l'image n'est pas calculé, on le met à 0.
    # Format : [[TP, FN], [FP, TN]]
    matrice = np.array([[TP, FN], 
                        [FP, 0]])

    plt.figure(figsize=(6, 5))
    
    # Le paramètre cmap="Blues" donne exactement les couleurs de votre exemple
    ax = sns.heatmap(matrice, annot=True, fmt="d", cmap="Blues", 
                     annot_kws={"size": 16}, cbar=True,
                     xticklabels=["Spot", "Background"], 
                     yticklabels=["Spot", "Background"])

    plt.title(f"Confusion Matrix - {nom_modele}", fontsize=14, pad=15)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    
    # Rotation des labels pour la lisibilité
    plt.yticks(rotation=0)
    
    nom_fichier = f"Confusion_Matrix_{nom_modele.replace('.pt', '')}.png"
    plt.savefig(nom_fichier, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 Image graphique générée : {nom_fichier}\n")


def evaluer_modele(nom_modele, type_modele="coco"):
    print(f"{'='*40}\n📊 ÉVALUATION DU MODÈLE : {nom_modele}\n{'='*40}")
    model = YOLO(nom_modele)
    
    if type_modele == "world":
        model.set_classes(["robot dog", "boston dynamics spot", "yellow quadruped robot"])

    TP, FP, FN = 0, 0, 0
    fichiers_images = [f for f in os.listdir(dossier_images) if f.endswith(('.jpg', '.png'))]

    for nom_img in fichiers_images:
        chemin_img = os.path.join(dossier_images, nom_img)
        nom_txt = os.path.splitext(nom_img)[0] + ".txt"
        chemin_txt = os.path.join(dossier_labels, nom_txt)

        if type_modele == "coco":
            result = model(chemin_img, conf=seuil_confiance, classes=[16], verbose=False)[0]
        else:
            result = model(chemin_img, conf=seuil_confiance, verbose=False)[0]
            
        boxes_predites = result.boxes.xyxy.cpu().numpy().tolist()
        
        boxes_reelles = []
        img_hauteur, img_largeur = result.orig_shape
        
        if os.path.exists(chemin_txt):
            with open(chemin_txt, 'r') as f:
                for ligne in f:
                    elements = ligne.strip().split()
                    if not elements: continue
                    x_c, y_c, w_norm, h_norm = map(float, elements[1:5])
                    w = w_norm * img_largeur
                    h = h_norm * img_hauteur
                    x1 = (x_c * img_largeur) - (w / 2)
                    y1 = (y_c * img_hauteur) - (h / 2)
                    boxes_reelles.append([x1, y1, x2 := x1 + w, y2 := y1 + h])

        pred_utilisees = []
        for box_reelle in boxes_reelles:
            match_trouve = False
            for i, box_pred in enumerate(boxes_predites):
                if i in pred_utilisees: continue
                
                if calculer_iou(box_reelle, box_pred) >= seuil_iou:
                    TP += 1
                    pred_utilisees.append(i)
                    match_trouve = True
                    break 
            
            if not match_trouve:
                FN += 1 
                
        FP += len(boxes_predites) - len(pred_utilisees)

    print(f"🎯 Vrais Positifs (TP) : {TP}")
    print(f"👻 Faux Positifs (FP)  : {FP}")
    print(f"🙈 Faux Négatifs (FN)  : {FN}")
    
    # Appel de la nouvelle fonction pour dessiner l'image
    dessiner_matrice(TP, FP, FN, nom_modele)

# Lancement
evaluer_modele("yolov8n.pt", type_modele="coco")
evaluer_modele("yolov8s-world.pt", type_modele="world")