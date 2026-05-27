# Données

Ce dossier est volontairement vide dans le dépôt git. Toutes les données
sont régénérables à partir de scripts. La structure se construit en
trois étapes, dans l'ordre :

## 1. Téléchargement des vidéos sources

Vidéos DARPA Subterranean Challenge et clips Boston Dynamics.

```bash
./scripts/02_download_darpa_videos.sh
```

Sortie : `data/videos/*.mp4` (environ 1,5 Go au total).

Prérequis : [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) installé.

## 2. Extraction des frames

Découpe les vidéos à 1 FPS et répartit les frames en splits
train/val/test selon la configuration de `scripts/03_extract_frames.py`.

```bash
python scripts/03_extract_frames.py
```

Sortie : `data/frames/{train,val,test}/*.jpg`.

Prérequis : `ffmpeg` installé sur la machine.

## 3. Annotation automatique par Grounding DINO

Applique Grounding DINO sur les frames extraites pour générer les
boîtes englobantes au format YOLO.

```bash
pip install -r scripts/requirements-annotate.txt
python scripts/04_annotate_dino.py
```

Sortie : `data/annotations_dino/{train,val,test}/{images,labels}/`.

Prérequis : GPU CUDA (l'annotation prend environ 15 min sur GPU,
plusieurs heures sur CPU).

## 4. Validation manuelle

Inspection visuelle et suppression des annotations incorrectes.

```bash
python scripts/05_verify_annotations.py --split train
python scripts/05_verify_annotations.py --split val
python scripts/05_verify_annotations.py --split test
```

Contrôles clavier : `[ESPACE]` garder, `[D]` supprimer, `[Q]` quitter.
La progression est sauvegardée, le script peut être interrompu et repris.

## Datasets Roboflow (phase archive)

La phase composite archivée (voir `archive/`) utilisait sept datasets
Roboflow publics. Pour les retélécharger :

```bash
export ROBOFLOW_API_KEY="votre_clé"
python scripts/01_download_roboflow.py
```

Sortie : `data/datasets/`.

Ces datasets ne sont pas nécessaires au pipeline principal.
