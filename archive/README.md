# Archive — Approche composite (mai 2026)

Ce dossier contient la **première approche** explorée pour la détection
du robot Spot par apprentissage profond, dans le cadre du stage
FILDARIANE. L'approche a été abandonnée après diagnostic d'un biais
structurel du dataset, et remplacée par le pipeline Teacher–Student
(Grounding DINO + YOLOv8m) qui constitue le pipeline principal du dépôt.

Les scripts et résultats sont conservés ici pour la **traçabilité du
diagnostic** et la valeur pédagogique du contre-exemple. Ils ne sont
pas conçus pour être réexécutés en l'état.

## Démarche initiale

L'idée était de construire un dataset composite en agrégeant les images
de Spot issues de **sept datasets Roboflow publics** (voir
`scripts/01_download_roboflow.py` à la racine du dépôt pour la liste).
Un modèle YOLOv8n a ensuite été entraîné sur ce composite, puis évalué
en parallèle de modèles pré-entraînés (YOLO-COCO, YOLO-World) et de
modèles communautaires hébergés sur Roboflow.

## Échec observé

Validation statique : excellente (matrices de confusion très favorables).
Validation dynamique sur les vidéos DARPA Subterranean Challenge :
**échec total**. Le modèle perd systématiquement sa cible dès que les
conditions s'écartent du dataset d'entraînement (éblouissement,
distance, occultations, flou de mouvement).

## Diagnostic

Audit rétrospectif du dataset : sur l'ensemble des données de la phase
composite, seules 18 images contenaient effectivement Spot, contre
353 images de décors purs. Le modèle a sur-appris les caractéristiques
des décors et non la signature géométrique du robot. Il n'avait pas
le vocabulaire visuel pour généraliser hors distribution.

Ce diagnostic a motivé le passage à une architecture Teacher–Student
exploitant directement les séquences vidéos cibles (DARPA Sub-T), avec
annotation automatique massive par Grounding DINO.

## Contenu du dossier

- `build_composite_test_set.py` — construction du set de test (6 images
  par dataset Roboflow).
- `build_composite_train_set.py` — construction du set d'entraînement
  (toutes images valides hors set de test).
- `convert_to_bw.py` — conversion du set de test en niveaux de gris
  pour la validation "signature structurelle".
- `benchmark/` — scripts d'évaluation comparative :
  - `benchmark_pretrained.py` — inférence visuelle de YOLO-COCO et
    YOLO-World sur les 30 images de test.
  - `confusion_pretrained.py` — matrices de confusion pour les deux
    modèles ci-dessus.
  - `confusion_roboflow_spot.py` — matrice pour l'API Roboflow
    `spot-8feoa`. Nécessite `ROBOFLOW_API_KEY`.
  - `confusion_roboflow_atlas.py` — matrice pour l'API Roboflow
    `spot-vs-atlas`. Nécessite `ROBOFLOW_API_KEY`.
  - `confusion_yolo_custom.py` — matrice du YOLO entraîné en local
    sur le composite (N&B), avec ajout manuel des vrais négatifs sur
    le set de décors vides.
- `configs/dataset_bw.yaml` — configuration YOLOv8 utilisée pour
  l'entraînement composite.
- `results/` — matrices de confusion finales (PNG).

## Reproduction

Non recommandée. Les chemins relatifs des scripts pointent vers une
arborescence (`datasets/`, `images/`, `labels/`) qui n'existe plus
dans le présent dépôt. Pour reproduire :

1. Recréer une arborescence `dataset_composite/` à part avec
   `datasets/` issu de `scripts/01_download_roboflow.py`.
2. Y exécuter dans l'ordre : `build_composite_test_set.py`, puis
   `build_composite_train_set.py`, éventuellement `convert_to_bw.py`.
3. Lancer un entraînement YOLOv8n classique avec
   `configs/dataset_bw.yaml`.
4. Évaluer avec les scripts du dossier `benchmark/`.
