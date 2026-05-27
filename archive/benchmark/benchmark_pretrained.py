from ultralytics import YOLO

dossier_test = "images/test"

print("🚀 Démarrage du Benchmark Local...\n")

# ==========================================
# TEST 1 : YOLO COCO (Le Généraliste)
# ==========================================
print("--- 1. ÉVALUATION YOLO COCO ---")
print("Chargement du modèle yolov8n.pt...")
model_coco = YOLO("yolov8n.pt")

print("Analyse des 30 images en cours...")
# On utilise project et name pour créer des dossiers de résultats propres
# conf=0.10 pour forcer le modèle à montrer ce qu'il "soupçonne"
model_coco(
    dossier_test, 
    conf=0.10, 
    project="resultats_benchmark", 
    name="coco_seuil_bas", 
    save=True,
    verbose=False, # Pour éviter de polluer le terminal
    classes = [16]
)
print("✅ Test COCO terminé.\n")

# ==========================================
# TEST 2 : YOLO-WORLD (Le Sémantique)
# ==========================================
print("--- 2. ÉVALUATION YOLO-WORLD ---")
print("Chargement du modèle yolov8s-world.pt...")
model_world = YOLO("yolov8s-world.pt")

# Définition de notre vocabulaire sur mesure
mots_cles = ["robot dog", "boston dynamics spot", "yellow quadruped robot"]
model_world.set_classes(mots_cles)

print(f"Recherche des classes : {mots_cles}")
print("Analyse des 30 images en cours...")
model_world(
    dossier_test, 
    conf=0.10, 
    project="resultats_benchmark", 
    name="world_seuil_bas", 
    save=True,
    verbose=False
)
print("✅ Test YOLO-World terminé.\n")

print("🎉 Benchmark terminé ! Allez voir le dossier 'resultats_benchmark/' pour comparer les images annotées.")