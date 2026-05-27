import os
import shutil
import yaml

dossiers_sources = "datasets"
dossier_test_img = "images/test"
dossier_train_img = "images/train"
dossier_train_lbl = "labels/train"
mots_cles = ["spot", "robodog", "dog", "yellow", "robot", "quadruped", "object"]

# Création automatique des dossiers s'ils n'existent pas
os.makedirs(dossier_train_img, exist_ok=True)
os.makedirs(dossier_train_lbl, exist_ok=True)

# 1. On liste les 30 images de test pour être sûr de ne pas les utiliser pour l'entraînement
images_test_existantes = set(os.listdir(dossier_test_img))

print("🚀 Démarrage de la création du dataset d'entraînement...\n")
images_ajoutees = 0

for nom_dossier in os.listdir(dossiers_sources):
    chemin_dossier = os.path.join(dossiers_sources, nom_dossier)
    if not os.path.isdir(chemin_dossier): continue
    
    chemin_yaml = os.path.join(chemin_dossier, "data.yaml")
    if not os.path.exists(chemin_yaml): continue
        
    # 2. Trouver l'ID de Spot dans les anciens dossiers
    with open(chemin_yaml, 'r') as f: data = yaml.safe_load(f)
    noms_classes = data.get('names', [])
    nc = data.get('nc', 0)
    id_spot = None
    
    if nc == 1 or len(noms_classes) == 1: id_spot = 0
    else:
        if isinstance(noms_classes, dict):
            for cid, cname in noms_classes.items():
                if any(m in str(cname).lower() for m in mots_cles): id_spot = cid; break
        elif isinstance(noms_classes, list):
            for i, cname in enumerate(noms_classes):
                if any(m in str(cname).lower() for m in mots_cles): id_spot = i; break
                
    if id_spot is None: continue

    chemin_images_train = os.path.join(chemin_dossier, "train", "images")
    chemin_labels_train = os.path.join(chemin_dossier, "train", "labels")
    if not os.path.exists(chemin_images_train): continue

    # 3. Filtrer et copier les images restantes
    for img_name in os.listdir(chemin_images_train):
        if not img_name.endswith(('.jpg', '.png', '.jpeg')): continue
        
        prefixe = nom_dossier[:5].replace(".", "")
        nouveau_nom_img = f"ds_{prefixe}_{img_name}"
        
        # PROTECTION : Si l'image est déjà dans le test, on l'ignore
        if nouveau_nom_img in images_test_existantes: continue
            
        nom_base = os.path.splitext(img_name)[0]
        chemin_txt = os.path.join(chemin_labels_train, nom_base + ".txt")
        nouveau_nom_txt = f"ds_{prefixe}_{nom_base}.txt"
        
        if os.path.exists(chemin_txt):
            lignes_conservees = []
            contient_spot = False
            with open(chemin_txt, 'r') as f:
                for ligne in f:
                    elements = ligne.strip().split()
                    if not elements: continue
                    # On force la classe à 0
                    if int(elements[0]) == id_spot:
                        lignes_conservees.append(f"0 {' '.join(elements[1:])}\n")
                        contient_spot = True
            
            # Si on a bien Spot, on sauvegarde l'image et l'étiquette corrigée
            if contient_spot:
                shutil.copy(os.path.join(chemin_images_train, img_name), os.path.join(dossier_train_img, nouveau_nom_img))
                with open(os.path.join(dossier_train_lbl, nouveau_nom_txt), 'w') as f:
                    f.writelines(lignes_conservees)
                images_ajoutees += 1

print(f"🎉 Terminé ! {images_ajoutees} images parfaites ont été ajoutées au dossier d'entraînement.")