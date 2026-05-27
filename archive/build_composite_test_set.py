import os
import random
import shutil
import yaml

dossiers_sources = "datasets"
dossier_out_img = "images/test"
dossier_out_lbl = "labels/test"
images_par_dataset = 6 

mots_cles = ["spot", "robodog", "dog", "yellow", "robot", "quadruped", "object"]

print("🚀 Démarrage de la récolte V2...\n")

for nom_dossier in os.listdir(dossiers_sources):
    chemin_dossier = os.path.join(dossiers_sources, nom_dossier)
    if not os.path.isdir(chemin_dossier):
        continue
        
    chemin_yaml = os.path.join(chemin_dossier, "data.yaml")
    if not os.path.exists(chemin_yaml):
        continue
        
    # 1. Analyser le data.yaml
    with open(chemin_yaml, 'r') as f:
        data = yaml.safe_load(f)
        
    noms_classes = data.get('names', [])
    nc = data.get('nc', 0)
    id_spot = None
    
    # Si le dataset ne contient qu'une seule classe, on assume que c'est Spot
    if nc == 1 or len(noms_classes) == 1:
        id_spot = 0
        nom_trouve = noms_classes[0] if isinstance(noms_classes, list) else noms_classes.get(0, "Inconnue")
        print(f"✅ Dataset '{nom_dossier}' -> 1 seule classe ('{nom_trouve}'), assumée comme Spot (ID 0).")
    else:
        # Sinon, on cherche avec les mots-clés
        if isinstance(noms_classes, dict):
            for class_id, class_name in noms_classes.items():
                if any(mot in str(class_name).lower() for mot in mots_cles):
                    id_spot = class_id
                    break
        elif isinstance(noms_classes, list):
            for i, class_name in enumerate(noms_classes):
                if any(mot in str(class_name).lower() for mot in mots_cles):
                    id_spot = i
                    break
            
    if id_spot is None:
        print(f"⚠️ Impossible de trouver Spot dans '{nom_dossier}'. Classes trouvées : {noms_classes}")
        continue
    elif nc != 1:
        print(f"✅ Dataset '{nom_dossier}' -> Spot trouvé à l'ID {id_spot}")
        
    # 2. Récupérer UNIQUEMENT les images contenant Spot
    chemin_images_train = os.path.join(chemin_dossier, "train", "images")
    chemin_labels_train = os.path.join(chemin_dossier, "train", "labels")
    
    if not os.path.exists(chemin_images_train):
        continue
        
    images_valides = []
    toutes_les_images = [img for img in os.listdir(chemin_images_train) if img.endswith(('.jpg', '.png', '.jpeg'))]
    
    for img_name in toutes_les_images:
        nom_base = os.path.splitext(img_name)[0]
        chemin_txt = os.path.join(chemin_labels_train, nom_base + ".txt")
        
        if os.path.exists(chemin_txt):
            with open(chemin_txt, 'r') as f:
                # Vérifie si au moins une ligne correspond à l'ID de Spot
                if any(int(ligne.strip().split()[0]) == id_spot for ligne in f if ligne.strip()):
                    images_valides.append(img_name)
                    
    if not images_valides:
        print("   -> Aucune image contenant Spot trouvée dans ce dataset.")
        continue
        
    # 3. Prendre 6 images au hasard parmi les valides
    images_choisies = random.sample(images_valides, min(images_par_dataset, len(images_valides)))
    
    # 4. Copier et corriger les labels
    for img_name in images_choisies:
        nom_base = os.path.splitext(img_name)[0]
        txt_name = nom_base + ".txt"
        
        chemin_img_src = os.path.join(chemin_images_train, img_name)
        chemin_txt_src = os.path.join(chemin_labels_train, txt_name)
        
        lignes_conservees = []
        with open(chemin_txt_src, 'r') as f:
            for ligne in f:
                elements = ligne.strip().split()
                if not elements: continue
                # Si c'est Spot, on force l'ID à 0
                if int(elements[0]) == id_spot:
                    lignes_conservees.append(f"0 {' '.join(elements[1:])}\n")
                    
        prefixe = nom_dossier[:5].replace(".", "")
        nouveau_nom_img = f"ds_{prefixe}_{img_name}"
        nouveau_nom_txt = f"ds_{prefixe}_{txt_name}"
        
        shutil.copy(chemin_img_src, os.path.join(dossier_out_img, nouveau_nom_img))
        with open(os.path.join(dossier_out_lbl, nouveau_nom_txt), 'w') as f:
            f.writelines(lignes_conservees)
            
    print(f"   -> {len(images_choisies)} images extraites et labels corrigés.\n")

print("🎉 Terminé ! Votre dataset de test est prêt.")