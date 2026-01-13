"""
Générateur de Données pour Visualisation
========================================
Ce script prend en entrée une instance (.tsp ou .in) et une solution (.out),
et génère un fichier JSON standardisé pour l'outil de visualisation Web.

Usage:
    py Generate_Visu_Data.py <instance_file> <solution_file>

Exemple:
    py Generate_Visu_Data.py ../data/Input/ali535.tsp ../data/Solutions/ali535_GRASP.out
"""

import sys
import os
import json
import math

# Ajout du path pour les modules src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from src.model.tsp_model import load_data, build_distance_matrix
except ImportError:
    # Fallback si lancé depuis un autre dossier ou structure non standard
    from Constructive_3 import load_data, build_distance_matrix

# --- MDS Logic (Intégrée pour être autonome ou via import) ---
def generate_mds_coords(distance_matrix):
    try:
        import numpy as np
        from sklearn.manifold import MDS
        print(" > Calcul MDS en cours...")
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4, max_iter=300)
        coords = mds.fit_transform(np.array(distance_matrix))
        return coords.tolist()
    except ImportError:
        print("ERREUR: scikit-learn manquant pour MDS. Installation requise : pip install scikit-learn")
        return []

def parse_solution_file(filename):
    """Lit le fichier .out (Format: ligne 1 = chemin, ligne 2 = coût)."""
    with open(filename, 'r') as f:
        lines = f.readlines()
        path = list(map(int, lines[0].strip().split()))
        cost = float(lines[1].strip())
    return path, cost

def generate_visu_json(instance_path, solution_path):
    print(f"Traitement : {instance_path} + {solution_path}")
    
    # 1. Charger l'instance
    # load_data retourne n, matrix (pour .in) ou n, matrix (pour .tsp via calcul)
    # MAIS ici on veut les COORDONNÉES si c'est un TSP, pas juste la matrice.
    # On va donc refaire un petit parsing spécifique ou améliorer load_data.
    # Pour faire simple et robuste, on gère les deux cas ici.
    
    is_tsp = instance_path.lower().endswith(".tsp")
    coords_2d = []
    
    if is_tsp:
        print(" > Format .tsp détecté : Lecture des coordonnées réelles.")
        # Parsing rapide pour récupérer les coords
        with open(instance_path, 'r') as f:
            lines = f.readlines()
        in_section = False
        for line in lines:
            if line.strip() == "NODE_COORD_SECTION":
                in_section = True
                continue
            if line.strip() == "EOF": break
            if in_section:
                parts = line.strip().split()
                if len(parts) >= 3:
                    coords_2d.append([float(parts[1]), float(parts[2])])
    else:
        print(" > Format .in détecté : Génération via MDS.")
        n, matrix = load_data(instance_path)
        coords_2d = generate_mds_coords(matrix)

    # 2. Charger la solution
    opt_path, opt_cost = parse_solution_file(solution_path)
    
    # 3. Préparer le JSON
    output_filename = solution_path.replace(".out", ".json")
    
    data = {
        "instance": os.path.basename(instance_path),
        "n_cities": len(coords_2d),
        "coordinates": coords_2d,
        "initial_path": list(range(len(coords_2d))), # Dummy path
        "initial_cost": 0, # Dummy
        "optimized_path": opt_path,
        "optimized_cost": opt_cost,
        "improvement": 0
    }
    
    with open(output_filename, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    print(f"✅ Fichier JSON généré : {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: py Generate_Visu_Data.py <instance> <solution>")
        # Test mode auto
        # print("Mode test...")
    else:
        generate_visu_json(sys.argv[1], sys.argv[2])
