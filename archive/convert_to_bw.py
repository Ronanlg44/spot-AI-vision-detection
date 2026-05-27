import cv2
import os
import shutil

# Dossiers sources
dossier_spot_img = "images/test"
dossier_spot_lbl = "labels/test"
dossier_decors = "decors_bruts"

# Dossiers destinations (propres et neufs)
dst_img = "images/test_bw"
dst_lbl = "labels/test_bw"

os.makedirs(dst_img, exist_ok=True)
os.makedirs(dst_lbl, exist_ok=True)

print("🖼️ 1. Conversion des 30 images de Spot (en gardant le nom d'origine)...")
spot_count = 0
if os.path.exists(dossier_spot_img):
    for img_name in os.listdir(dossier_spot_img):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Conversion image
            img = cv2.imread(os.path.join(dossier_spot_img, img_name))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(dst_img, img_name), gray)
                spot_count += 1
                
                # Copie automatique et sécurisée du label (.txt) associé
                nom_base = os.path.splitext(img_name)[0]
                fichier_txt = nom_base + ".txt"
                chemin_txt_src = os.path.join(dossier_spot_lbl, fichier_txt)
                if os.path.exists(chemin_txt_src):
                    shutil.copy(chemin_txt_src, os.path.join(dst_lbl, fichier_txt))
else:
    print("⚠️ Erreur : Le dossier images/test n'existe pas !")

print(f"🕳️ 2. Conversion des décors bruts (grottes/entrepôts)...")
decor_count = 0
if os.path.exists(dossier_decors):
    for img_name in os.listdir(dossier_decors):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(dossier_decors, img_name))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(dst_img, img_name), gray)
                decor_count += 1
else:
    print("⚠️ Erreur : Le dossier decors_bruts n'existe pas !")

print(f"\n🎉 BASE DE TEST PRÊTE !")
print(f"👉 {spot_count} images de Spot converties + leurs étiquettes copiées.")
print(f"👉 {decor_count} images de fonds vides converties.")