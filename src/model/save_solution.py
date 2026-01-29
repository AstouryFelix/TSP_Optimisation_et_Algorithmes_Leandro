import os

def save_solution(filename, path, cost, zero_based=True):
    """
    Sauvegarde la solution au format demandé : instance_method.out
    
    Format du fichier de sortie :
    - Ligne 1 : sommets séparés par des espaces (indices 1-based)
    - Ligne 2 : coût total du cycle
    
    Args:
        filename: Chemin du fichier de sortie
        path: Liste des sommets dans l'ordre de visite
        cost: Coût total du cycle
        zero_based: Si True, les indices dans path sont 0-based et seront convertis en 1-based
    """
    # Convertir en indices 1-based si nécessaire
    if zero_based:
        path_output = [p + 1 for p in path]
    else:
        path_output = path
    
    # Assurer que le dossier parent existe (si spécifié)
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    with open(filename, "w") as f:
        f.write(" ".join(map(str, path_output)) + "\n")
        f.write(str(int(cost)) + "\n")
    
    print(f"Solution sauvegardée : {filename}")