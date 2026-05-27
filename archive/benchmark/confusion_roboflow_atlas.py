import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from roboflow import Roboflow

# ==========================================
# CONFIGURATION
# ==========================================
CLE_API = os.environ.get("ROBOFLOW_API_KEY")
if not CLE_API:
    raise SystemExit(
        "Set ROBOFLOW_API_KEY in your environment first:\n"
        "  export ROBOFLOW_API_KEY=\"your_key\""
    )

dossier_images = "images/test"
dossier_labels = "labels/test"
seuil_iou = 0.4

# Modèle 2 : Entraîné pour différencier Spot et Atlas
rf = Roboflow(api_key=CLE_API)
project = rf.workspace("university-dnc5w").project("spot-vs-atlas")
model = project.version(1).model 

def calculer_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea) if float(boxAArea + boxBArea - interArea) > 0 else 0

def dessiner_matrice(TP, FP, FN, nom_fichier, titre):
    matrice = np.array([[TP, FN], [FP, 0]])
    plt.figure(figsize=(6, 5))
    # En violet pour le différencier facilement du premier modèle
    sns.heatmap(matrice, annot=True, fmt="d", cmap="Purples", annot_kws={"size": 16}, 
                cbar=True, xticklabels=["Spot", "Background"], yticklabels=["Spot", "Background"])
    plt.title(titre, fontsize=14, pad=15)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.yticks(rotation=0)
    plt.savefig(nom_fichier, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 Graphique généré : {nom_fichier}\n")

print(f"{'='*40}\n📊 ÉVALUATION : MODÈLE ANTI-CONFUSION (spot-vs-atlas)\n{'='*40}")

TP, FP, FN = 0, 0, 0
fichiers_images = [f for f in os.listdir(dossier_images) if f.endswith(('.jpg', '.png'))]

for i, nom_img in enumerate(fichiers_images):
    print(f"Analyse {i+1}/{len(fichiers_images)} : {nom_img}...")
    chemin_img = os.path.join(dossier_images, nom_img)
    nom_txt = os.path.splitext(nom_img)[0] + ".txt"
    chemin_txt = os.path.join(dossier_labels, nom_txt)

    img = cv2.imread(chemin_img)
    img_hauteur, img_largeur = img.shape[:2]

    # Requête API avec un seuil à 25%
    result = model.predict(chemin_img, confidence=10).json()

    # Sauvegarde l'image avec les boîtes dessinées par l'API dans un dossier de résultats
    model.predict(chemin_img, confidence=10).save(os.path.join("runs/detect/resultats_benchmark/api_SpotAtlas", nom_img))
    
    boxes_predites = []
    for pred in result.get("predictions", []):
        # Attention : On s'assure de ne récupérer que les prédictions "Spot" (souvent classe 'spot' ou '1')
        # L'API renvoie le nom de la classe en texte dans pred['class']
        if "atlas" not in pred['class'].lower(): 
            w, h = pred['width'], pred['height']
            x1, y1 = pred['x'] - (w / 2), pred['y'] - (h / 2)
            boxes_predites.append([x1, y1, x1 + w, y1 + h])

    boxes_reelles = []
    if os.path.exists(chemin_txt):
        with open(chemin_txt, 'r') as f:
            for ligne in f:
                elements = ligne.strip().split()
                if not elements: continue
                x_c, y_c, w_norm, h_norm = map(float, elements[1:5])
                w, h = w_norm * img_largeur, h_norm * img_hauteur
                x1, y1 = (x_c * img_largeur) - (w / 2), (y_c * img_hauteur) - (h / 2)
                boxes_reelles.append([x1, y1, x1 + w, y1 + h])

    pred_utilisees = []
    for box_reelle in boxes_reelles:
        match_trouve = False
        for j, box_pred in enumerate(boxes_predites):
            if j in pred_utilisees: continue
            if calculer_iou(box_reelle, box_pred) >= seuil_iou:
                TP += 1
                pred_utilisees.append(j)
                match_trouve = True
                break 
        if not match_trouve: FN += 1 
            
    FP += len(boxes_predites) - len(pred_utilisees)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
rappel = TP / (TP + FN) if (TP + FN) > 0 else 0

print(f"\n🎯 Vrais Positifs : {TP} | 👻 Faux Positifs : {FP} | 🙈 Faux Négatifs : {FN}")
print(f"Précision : {precision:.1%} | Rappel : {rappel:.1%}")
dessiner_matrice(TP, FP, FN, "Confusion_Matrix_API_Atlas.png", "API - Spot VS Atlas")
