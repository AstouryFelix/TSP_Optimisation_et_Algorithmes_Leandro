"""
Question 3 : Heuristique Constructive (Nearest Neighbor)
========================================================
Ce module sert de base pour le projet TSP.
Il contient :
- Les fonctions de lecture de fichier (.in et .tsp)
- Le calcul de la matrice de distances (Euclidien / GEO)
- L'algorithme Constructif (Plus proche voisin)
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.model.load_data import *
from src.model.total_cost import *
from src.model.export_to_json import *
from src.model.save_solution import *

def constructive_nearest_neighbor(n, matrix, start_node=0):
    """Algorithme glouton du plus proche voisin."""
    unvisited = set(range(n))
    current_node = start_node
    path = [current_node]
    unvisited.remove(current_node)
    
    while unvisited:
        nearest_node = None
        min_dist = float('inf')
        
        for neighbor in unvisited:
            dist = matrix[current_node][neighbor]
            if dist < min_dist:
                min_dist = dist
                nearest_node = neighbor
        
        current_node = nearest_node
        path.append(current_node)
        unvisited.remove(current_node)
            
    return path


# --- MAIN (TEST Q3) ---
if __name__ == "__main__":
    print("=== TEST QUESTION 3 : NEAREST NEIGHBOR ===")
    
    # Test sur 100.in (depuis la racine du projet ou dossier courant)
    # On cherche le fichier dans les emplacements possibles
    possible_paths_100 = [
        "instances/constructive/100.in",
        "../instances/constructive/100.in",
        "../../instances/constructive/100.in"
    ]
    
    file1 = None
    for p in possible_paths_100:
        if os.path.exists(p):
            file1 = p
            break
            
    if file1:
        print(f"\nTraitement de {file1}...")
        n, mat = load_data(file1)
        path = constructive_nearest_neighbor(n, mat)
        cost = calculate_total_cost(path, mat)
        print(f"Cout NN: {cost}")
        
        # Sauvegarde
        base_name = os.path.basename(file1).replace(".in","").replace(".tsp","")
        outfile = f"Solutions/{base_name}_constructive.out"
        save_solution(outfile, path, cost)
        export_to_json(file1, mat, path, cost, "_constructive")
    else:
        print("Fichier 100.in introuvable dans instances/constructive/")

    # Test sur ali535.tsp
    possible_paths_ali = [
        "instances/constructive/ali535.tsp",
        "../instances/constructive/ali535.tsp",
        "../../instances/constructive/ali535.tsp"
    ]
    
    file2 = None
    for p in possible_paths_ali:
        if os.path.exists(p):
            file2 = p
            break

    if file2:
        print(f"\nTraitement de {file2}...")
        n, mat = load_data(file2)
        path = constructive_nearest_neighbor(n, mat)
        cost = calculate_total_cost(path, mat)
        print(f"Cout NN: {cost}")
        
        base_name = os.path.basename(file2).replace(".in","").replace(".tsp","")
        outfile = f"Solutions/{base_name}_constructive.out"
        save_solution(outfile, path, cost)
        export_to_json(file2, mat, path, cost, "_constructive")
    else:
        print("Fichier ali535.tsp introuvable dans instances/constructive/")