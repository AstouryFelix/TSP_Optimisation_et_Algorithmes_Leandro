"""
Question 3 : Heuristique Constructive (Nearest Neighbor)
========================================================
Ce module sert de base pour le projet TSP.
Il contient :
- Les fonctions de lecture de fichier (.in et .tsp)
- Le calcul de la matrice de distances (Euclidien / GEO)
- L'algorithme Constructif (Plus proche voisin)
"""

import os
from Tools.load_data      import *
from Tools.total_cost     import *
from Tools.export_to_json import *
from Tools.save_solution  import *

# --- ALGORITHME Q3 : NEAREST NEIGHBOR ---

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
    
    # Test sur 100.in
    file1 = "../data/Input/100.in"
    if os.path.exists(file1):
        print(f"\nTraitement de {file1}...")
        n, mat = load_data(file1)
        path = constructive_nearest_neighbor(n, mat)
        cost = calculate_total_cost(path, mat)
        print(f"Cout NN: {cost}")
        save_solution("../data/Solutions/100_constructive.out", path, cost)
        export_to_json(file1, mat, path, cost, "_constructive")

    # Test sur ali535.tsp
    file2 = "../data/Input/ali535.tsp"
    if os.path.exists(file2):
        print(f"\nTraitement de {file2}...")
        n, mat = load_data(file2)
        path = constructive_nearest_neighbor(n, mat)
        cost = calculate_total_cost(path, mat)
        print(f"Cout NN: {cost}")
        save_solution("../data/Solutions/ali535_constructive.out", path, cost)
        export_to_json(file2, mat, path, cost, "_constructive")