"""
=============================================================================
PROJET : VISION ENGINE - EMBEDDINGS (CLIP)
DATE   : 19 Février 2026
DEV    : Léo
=============================================================================

LOG :
--------
[19/02 19:30] Init du moteur de vision avec la lib transformers.
[19/02 20:15] Choix de CLIP vit-base-patch32. Le modele est vieux mais super 
              léger, ça tourne bien même sans gros GPU.
[19/02 21:00] Premier test : crash total. Le modele plantait en cherchant des 
              'input_ids' (texte) alors que je lui passais une image.
              Fix : utilisation stricte de get_image_features().
[19/02 22:30] Deuxieme crash au moment du detach(). L'output n'etait pas 
              toujours un tenseur pur mais un objet HuggingFace custom. 
              Ajout d'un deballage conditionnel.
=============================================================================
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

print("Chargement des poids du modele CLIP (vit-base-patch32)...")
MODEL_ID = "openai/clip-vit-base-patch32" 

# Chargement lourd au demarrage. On le fait en global pour eviter 
# de recharger les poids a chaque appel de la fonction.
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID)
print("Modele charge en memoire.")

def get_image_embedding(image_path):
    try:
        # Forcage en RGB sinon les images PNG avec transparence (RGBA) font crasher le processor
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        # Mode inference pur, on coupe le calcul du gradient pour economiser la RAM
        with torch.no_grad():
            # Bypass de la methode forward classique pour eviter l'erreur d'input_ids manquant
            outputs = model.get_image_features(pixel_values=inputs['pixel_values'])
            
        # Le fix de 22h30 :
        # HuggingFace aime bien renvoyer des classes 'BaseModelOutputWithPooling' au lieu de tenseurs.
        # Si c'est le cas, le vrai tenseur est cache a l'index 0.
        tensor = outputs[0] if not isinstance(outputs, torch.Tensor) else outputs
        
        # Passage CPU obligatoire avant de convertir en numpy, puis aplatissement
        vector = tensor.detach().cpu().numpy().flatten().tolist()
        
        return vector
        
    except Exception as e:
        print(f"Erreur fatale sur l'image {image_path} : {e}")
        return None


# --- SCRIPT DE DEBUG INDEPENDANT ---
if __name__ == "__main__":
    # Permet de tester le moteur de vision en isolation sans lancer toute l'usine a gaz
    print("\n--- Diagnostic de l'extracteur de features ---")
    
    # Generation d'une mire noire a la volee pour eviter de chercher un fichier de test
    img_test = Image.new('RGB', (200, 200), color = 'black')
    img_test.save("test_noir.jpg")
    
    vecteur = get_image_embedding("test_noir.jpg")
    
    if vecteur:
        print(f"Extraction reussie. Taille du vecteur : {len(vecteur)} dimensions.")
        print(f"Sample (5 premieres valeurs) : {vecteur[:5]}")
    else:
        print("Echec de l'extraction.")