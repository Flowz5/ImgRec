"""
=============================================================================
PROJET : VISION ENGINE - CHROMA DB MEMORY
DATE   : 19 Février 2026
DEV    : Léo
=============================================================================

LOG :
--------
[19/02 21:00] Creation du connecteur ChromaDB pour les vecteurs d'images.
[19/02 21:45] Setup de la collection avec distance Cosine (plus opti pour les embeddings).
[19/02 22:30] Gros bug sur l'insertion. Le modele renvoyait des tenseurs 4D.
              Fix: passage par numpy.flatten() pour aplatir tout ca en liste 1D.
[19/02 23:15] Implementation d'un k-NN basique (k=3) avec Counter pour la prediction.
=============================================================================
"""

import chromadb
import uuid
import os
import numpy as np
from collections import Counter

# Resolution dynamique du chemin pour eviter les crashs selon d'ou on lance le script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_vision")

# Initialisation de la BDD persistante
client = chromadb.PersistentClient(path=DB_PATH)

# Definition de l'espace HNSW en "cosine". Indispensable pour que la 
# similarite entre les vecteurs d'images ait du sens mathematiquement.
collection = client.get_or_create_collection(
    name="vision_memory", 
    metadata={"hnsw:space": "cosine"}
)

def get_nb_images():
    # Simple helper pour le compteur de l'UI
    return collection.count()

def add_to_memory(image_path, vector, label):
    # Securite anti-crash au cas ou l'extracteur de features plante en amont
    if vector is None or len(vector) == 0:
        print("Erreur : Le vecteur est vide !")
        return

    try:
        # Le fix de 22h30 : 
        # Les modeles de vision recrachent souvent des tenseurs imbriques.
        # Numpy aplatit tout ca en une simple liste de floats 1D lisible par Chroma.
        clean_vector = np.array(vector).flatten().astype(float).tolist()
        
        collection.add(
            embeddings=[clean_vector],
            metadatas=[{"label": label, "path": image_path}],
            ids=[str(uuid.uuid4())] # Generateur d'ID unique natif
        )
        print(f"Enregistre en base : {label}")
    except Exception as e:
        print(f"Erreur sauvegarde : {e}")

def predict(vector, k=3):
    count = collection.count()
    if count == 0:
        return "Inconnu (Memoire vide)"
    
    # Il faut absolument aplatir le vecteur de requete de la meme maniere
    clean_vector = np.array(vector).flatten().tolist()
    
    # On recupere les k plus proches voisins (ou moins si on a peu d'images)
    results = collection.query(
        query_embeddings=[clean_vector],
        n_results=min(k, count)
    )
    
    # Extraction des labels depuis la sous-liste des metadatas
    labels = [meta["label"] for meta in results["metadatas"][0]]
    
    # Logique K-NN (K-Nearest Neighbors) : on fait un vote a la majorite.
    # most_common(1) renvoie un tuple type [('Chat', 2)], on extrait juste la string.
    return Counter(labels).most_common(1)[0][0]