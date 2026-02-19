# 👁️ MonIAVision : Apprentissage Continu & Vision TUI

Une application de classification d'images "Human-in-the-loop" capable d'apprendre en temps réel. Plus vous l'utilisez, plus elle devient intelligente grâce à sa mémoire vectorielle persistante.

## 🚀 Fonctionnalités

* **Apprentissage Continu** : L'IA ne se contente pas de prédire ; elle apprend de vos corrections instantanément et ne les oublie jamais.
* **Moteur CLIP** : Utilise le modèle `openai/clip-vit-base-patch32` pour une compréhension visuelle de pointe.
* **Mémoire Vectorielle** : Propulsé par **ChromaDB** pour stocker et comparer les images mathématiquement.
* **Interface Terminale (TUI)** : Une expérience utilisateur élégante et colorée réalisée avec **Rich**.
* **Gestion Robuste** : Correction des erreurs de formats de données (`float` vs `list`) et des conflits d'entrée de modèles.

## 🛠️ Stack Technique

* **Langage** : Python 3.12 (optimisé pour Fedora/Hyprland).
* **IA & Mathématiques** : `torch`, `transformers` (CLIP), `numpy`.
* **Base de données** : `chromadb` (moteur vectoriel SQLite).
* **UI & Feedback** : `rich` pour le terminal, `imv` pour l'affichage d'images.

## 📥 Installation

1. **Prérequis** : Assurez-vous d'avoir Python 3.12 et le visionneur `imv` installés.
```bash
sudo dnf install imv

```


2. **Installation des dépendances** :
```bash
pip install torch torchvision transformers chromadb rich pillow numpy

```


3. **Structure du projet** :
```text
.
├── data/               # Mémoire de l'IA (ChromaDB)
├── images_test/        # Vos photos (chiens, chats, etc.)
├── main.py             # Interface et boucle principale
├── memory_vision.py    # Gestionnaire de la base de données
└── vision_engine.py    # Moteur d'extraction visuelle

```



## 🎮 Utilisation

1. Glissez vos images dans le dossier `images_test/`.
2. Lancez le programme :
```bash
python main.py

```


3. **Le cycle d'apprentissage** :
* L'IA tente une prédiction sur l'image choisie.
* Si elle se trompe (ou si elle ne connaît pas encore l'objet), donnez-lui la bonne réponse.
* Le compteur "Images en mémoire" augmente : l'IA est maintenant capable de reconnaître des objets similaires !



## 🧠 Notes Techniques

Contrairement au Deep Learning classique, ce projet n'utilise pas de "fine-tuning" (ré-entraînement lourd). Il utilise la **recherche de similarité cosinus**. L'IA transforme chaque image en un vecteur de 512 dimensions et les compare dans un espace mathématique pour trouver les correspondances les plus proches.