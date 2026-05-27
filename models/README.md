# Modèles

Le checkpoint final `best.pt` (YOLOv8m fine-tuné, ~52 Mo) n'est pas
versionné dans git. Il est hébergé sur Hugging Face Hub.

## Téléchargement

```bash
wget https://huggingface.co/ronanlg44/spot-yolov8m/resolve/main/best.pt -O models/best.pt
```

## Utilisation

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict("path/to/image.jpg")
```

## Caractéristiques du modèle

- **Architecture** : YOLOv8m (medium, ~25M paramètres)
- **Classe détectée** : Spot (Boston Dynamics quadruped robot)
- **Image size** : 640 × 640
- **Entraînement** : 50 epochs, batch 16, optimizer auto (SGD par défaut)
- **Dataset** : frames DARPA Subterranean Challenge annotées par
  Grounding DINO, validées manuellement (voir `notebooks/train_yolov8m.ipynb`)
- **Hardware d'entraînement** : GPU NVIDIA, environ 30min
