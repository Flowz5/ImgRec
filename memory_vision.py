import chromadb
import uuid
import os
import numpy as np
from collections import Counter

# Chemin vers la base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_vision")

# Initialisation
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name="vision_memory", 
    metadata={"hnsw:space": "cosine"}
)

def get_nb_images():
    return collection.count()

def add_to_memory(image_path, vector, label):
    # Sécurité si l'extracteur renvoie du vide
    if vector is None or len(vector) == 0:
        print("⚠️ Erreur : Le vecteur est vide !")
        return

    try:
        # On utilise numpy pour écraser tous les crochets superflus (les fameux [[[[ ]]]])
        #
        clean_vector = np.array(vector).flatten().astype(float).tolist()
        
        collection.add(
            embeddings=[clean_vector],
            metadatas=[{"label": label, "path": image_path}],
            ids=[str(uuid.uuid4())]
        )
        print(f"💾 Enregistré en base : {label}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde : {e}")

def predict(vector, k=3):
    count = collection.count()
    if count == 0:
        return "Inconnu (Mémoire vide)"
    
    # On aplatit aussi pour la recherche
    clean_vector = np.array(vector).flatten().tolist()
    
    results = collection.query(
        query_embeddings=[clean_vector],
        n_results=min(k, count)
    )
    labels = [meta["label"] for meta in results["metadatas"][0]]
    return Counter(labels).most_common(1)[0][0]