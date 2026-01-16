import os

def save_solution(filename, path, cost):
    """Sauvegarde la solution dans data/Solutions."""
    # Assurer que le dossier existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(" ".join(map(str, path)) + "\n")
        f.write(str(cost) + "\n")
    print(f"Solution sauvegardée : {filename}")