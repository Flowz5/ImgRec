"""
=============================================================================
PROJET : VISION ENGINE CLI (FEW-SHOT LEARNING)
DATE   : 19 Février 2026
DEV    : Léo
=============================================================================

LOG :
--------
[19/02 21:00] Init de la boucle de feedback visuel.
[19/02 21:30] Integration de subprocess pour ouvrir l'image en direct avec 'imv'. 
              Ca evite de devoir alt-tab pour voir de quoi le script parle.
[19/02 22:15] Fix du compteur de souvenirs qui restait bloqué à 0. 
              Il fallait le refresh à l'intérieur de la boucle while.
[19/02 22:45] Ajout du flow d'apprentissage actif. Si le modele se trompe, 
              on force le bon label en base pour corriger les prochains vecteurs.
=============================================================================
"""

import os
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

# Imports des modules locaux de traitement d'image et de BDD vectorielle
from vision_engine import get_image_embedding
from memory_vision import add_to_memory, predict, get_nb_images

console = Console()
DOSSIER_IMAGES = "images_test"

def main():
    while True:
        # Refresh dynamique à chaque tour de boucle
        # Sinon l'UI affiche toujours le même nombre même après un apprentissage
        nb_souvenirs = get_nb_images()
        
        console.clear()
        console.print(Panel.fit(f"[bold blue]👁️ IA Vision Continue[/bold blue]\n[green]Images en mémoire : {nb_souvenirs}[/green]"))
        
        # Scan du dossier d'input
        images = [f for f in os.listdir(DOSSIER_IMAGES) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            console.print("[yellow]Glisse des images dans 'images_test/'[/yellow]")
            break

        # Affichage propre de la liste des fichiers
        table = Table()
        table.add_column("N°", style="cyan")
        table.add_column("Fichier")
        for i, name in enumerate(images):
            table.add_row(str(i+1), name)
        console.print(table)

        choix = Prompt.ask("\nChoisis une image (ou 'q')")
        if choix.lower() == 'q': break
        
        img_path = os.path.join(DOSSIER_IMAGES, images[int(choix)-1])
        
        # Astuce UX : on ouvre l'image en arriere-plan avec 'imv' (ou un autre viewer leger)
        # On redirige stdout/stderr vers DEVNULL pour pas polluer le terminal
        proc = subprocess.Popen(["imv", img_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Extraction du vecteur (le gros du temps de calcul est ici)
        with console.status("Analyse..."):
            vecteur = get_image_embedding(img_path)
        
        # Comparaison Cosine avec ChromaDB
        prediction = predict(vecteur)
        console.print(Panel(f"Je pense que c'est : [bold]{prediction}[/bold]"))

        # Logique de Few-Shot Learning
        # Si la base est vide (0 souvenir) ou que la prediction est fausse, on demande le vrai label
        if nb_souvenirs == 0 or not Confirm.ask("Est-ce correct ?"):
            vrai_label = Prompt.ask("C'est quoi alors ?")
            # On stocke le vecteur avec le label corrigé pour affiner la future precision
            add_to_memory(img_path, vecteur, vrai_label.capitalize())
        else:
            # Si c'est juste, on renforce ce cluster en ajoutant cette nouvelle variante de l'image
            add_to_memory(img_path, vecteur, prediction)

        # On kill le process du viewer d'image proprement avant de passer a la suite
        if proc: proc.terminate()
        Prompt.ask("\n[dim]Appuie sur Entrée pour continuer...[/dim]")

if __name__ == "__main__":
    main()