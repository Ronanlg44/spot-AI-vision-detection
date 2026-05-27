import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. On définit vos compteurs réels basés sur votre test
# Remplissez ici le nombre EXACT d'images de décors vides que vous avez téléchargées
NB_TOTAL_IMAGES_DECORS_VIDES = 353 

# Chiffres officiels de votre terminal :
true_spot_predicted_spot = 17   # Vrais Positifs
true_spot_predicted_bg = 2      # Faux Négatifs (Spot ratés)
faux_positifs = 1              # L'IA a vu Spot dans le décor

# Le calcul magique que YOLO refusait de faire :
vrais_negatifs = NB_TOTAL_IMAGES_DECORS_VIDES - faux_positifs

# 2. Construction de la matrice brute
matrice_brute = np.array([
    [true_spot_predicted_spot, faux_positifs],
    [true_spot_predicted_bg, vrais_negatifs]
])

# 3. Construction de la matrice NORMALISÉE (par colonne/vérité)
# Colonne 1 (True Spot) divisée par 19 (17+2)
# Colonne 2 (True Background) divisée par le nombre total de décors
col1 = matrice_brute[:, 0] / (true_spot_predicted_spot + true_spot_predicted_bg)
col2 = matrice_brute[:, 1] / NB_TOTAL_IMAGES_DECORS_VIDES
matrice_normalisee = np.stack([col1, col2], axis=1)

# 4. Génération du graphique propre
categories = ['Spot', 'background']
plt.figure(figsize=(8, 6))
sns.heatmap(matrice_normalisee, annot=matrice_brute, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories, cbar=True, vmin=0, vmax=1)

# Ajout des pourcentages en texte superposé pour le rapport
for i in range(2):
    for j in range(2):
        plt.text(j + 0.5, i + 0.7, f"({matrice_normalisee[i, j]*100:.1f}%)", 
                 ha='center', va='center', color='black' if matrice_normalisee[i, j] < 0.5 else 'white')

plt.title('Confusion Matrix - Yolo v8n trained - B&W Dataset')
plt.ylabel('Predicted')
plt.xlabel('True')
plt.tight_layout()

# Sauvegarde de l'image
plt.savefig('runs/detect/resultats_benchmark/Confusion_Matrix_YoloLocal.png', dpi=300)
print("🎉 Superbe ! Le fichier 'Confusion_Matrix_YoloLocal.png' a été généré.")