import os
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from vision_engine import get_image_embedding
from memory_vision import add_to_memory, predict, get_nb_images

console = Console()
DOSSIER_IMAGES = "images_test"

def main():
    while True:
        # ON RAFRAÎCHIT LE COMPTEUR ICI
        nb_souvenirs = get_nb_images()
        
        console.clear()
        console.print(Panel.fit(f"[bold blue]👁️ IA Vision Continue[/bold blue]\n[green]Images en mémoire : {nb_souvenirs}[/green]"))
        
        images = [f for f in os.listdir(DOSSIER_IMAGES) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            console.print("[yellow]Glisse des images dans 'images_test/'[/yellow]")
            break

        table = Table()
        table.add_column("N°", style="cyan")
        table.add_column("Fichier")
        for i, name in enumerate(images):
            table.add_row(str(i+1), name)
        console.print(table)

        choix = Prompt.ask("\nChoisis une image (ou 'q')")
        if choix.lower() == 'q': break
        
        img_path = os.path.join(DOSSIER_IMAGES, images[int(choix)-1])
        
        # Ouverture image
        proc = subprocess.Popen(["imv", img_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        with console.status("Analyse..."):
            vecteur = get_image_embedding(img_path)
        
        prediction = predict(vecteur)
        console.print(Panel(f"Je pense que c'est : [bold]{prediction}[/bold]"))

        # Apprentissage
        if nb_souvenirs == 0 or not Confirm.ask("Est-ce correct ?"):
            vrai_label = Prompt.ask("C'est quoi alors ?")
            add_to_memory(img_path, vecteur, vrai_label.capitalize())
        else:
            add_to_memory(img_path, vecteur, prediction)

        if proc: proc.terminate()
        Prompt.ask("\n[dim]Appuie sur Entrée pour continuer...[/dim]")

if __name__ == "__main__":
    main()