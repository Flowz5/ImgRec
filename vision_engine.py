import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Chargement du modèle (Il va se télécharger automatiquement la première fois)
print("👁️ Chargement des 'yeux' de l'IA (Modèle CLIP)...")
# On utilise une version légère et rapide de CLIP
MODEL_ID = "openai/clip-vit-base-patch32" 

processor = CLIPProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID)
print("✅ Yeux opérationnels !")

def get_image_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            # On utilise get_image_features pour éviter l'erreur input_ids
            outputs = model.get_image_features(pixel_values=inputs['pixel_values'])
            
        # --- LA CORRECTION "DÉBALLAGE" ---
        # Si outputs est la "boîte" (BaseModelOutputWithPooling), on prend son premier élément [0]
        # Sinon, si c'est déjà un tenseur, on le garde tel quel.
        tensor = outputs[0] if not isinstance(outputs, torch.Tensor) else outputs
        
        # Maintenant on peut détacher et aplatir sans erreur
        vector = tensor.detach().cpu().numpy().flatten().tolist()
        
        return vector
        
    except Exception as e:
        print(f"❌ Erreur lors de la lecture de l'image : {e}")
        return None

# --- PETIT CRASH TEST ---
if __name__ == "__main__":
    # Ce code ne s'exécute que si on lance ce fichier directement
    print("\n--- Test de l'extracteur ---")
    
    # 1. Crée une image noire de test temporaire juste pour voir si le code tourne
    img_test = Image.new('RGB', (200, 200), color = 'black')
    img_test.save("test_noir.jpg")
    
    # 2. On la fait lire par l'IA
    vecteur = get_image_embedding("test_noir.jpg")
    
    if vecteur:
        print(f"Succès ! L'image a été transformée en une liste de {len(vecteur)} nombres.")
        print(f"Voici les 5 premiers nombres du vecteur : {vecteur[:5]}")