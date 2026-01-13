"""
Recherche Locale : 2-Opt
========================
"""

import sys
import os

# Ajout du dossier parent au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.tsp_model import load_data, calculate_total_cost, save_solution, export_to_json

def local_search_2opt(path, matrix):
    """Améliore un chemin existant en utilisant l'opérateur "2-opt"."""
    n = len(path)
    improved = True
    best_path = path[:] 
    
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n): 
                if j == n - 1 and i == 0: continue
                
                u, v = best_path[i], best_path[i+1]
                x, y = best_path[j], best_path[(j + 1) % n]
                
                cost_current = matrix[u][v] + matrix[x][y]
                cost_new = matrix[u][x] + matrix[v][y]
                
                if cost_new < cost_current:
                    best_path[i+1 : j+1] = best_path[i+1 : j+1][::-1]
                    improved = True
    return best_path

if __name__ == "__main__":
    print("=== TEST Q4: 2-OPT ===")
    # Pour tester, on a besoin d'une solution initiale.
    # On importe le NN.
    from src.constructive.nearest_neighbor import constructive_nearest_neighbor
    
    filename = "../../data/Input/100.in"
    if os.path.exists(filename):
        n, mat = load_data(filename)
        init_path = constructive_nearest_neighbor(n, mat)
        opt_path = local_search_2opt(init_path, mat)
        print(f"Cout Optimisé: {calculate_total_cost(opt_path, mat)}")
