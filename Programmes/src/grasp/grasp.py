"""
Métaheuristique : GRASP
=======================
"""

import sys
import os
import random

# Ajout du dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.tsp_model import load_data, calculate_total_cost, save_solution, export_to_json
from src.local_search.two_opt import local_search_2opt

def constructive_randomized_nearest_neighbor(n, matrix, alpha=1, start_node=0):
    """Variante randomisée du NN (Phase 1)."""
    unvisited = set(range(n))
    current_node = start_node
    path = [current_node]
    unvisited.remove(current_node)
    
    while unvisited:
        candidates = []
        for neighbor in unvisited:
            dist = matrix[current_node][neighbor]
            candidates.append((dist, neighbor))
        
        candidates.sort(key=lambda x: x[0])
        rcl_size = min(alpha, len(candidates))
        rcl = candidates[:rcl_size]
        
        _, chosen_node = random.choice(rcl)
        current_node = chosen_node
        path.append(current_node)
        unvisited.remove(current_node)
    return path

def run_grasp(n, matrix, max_iterations=20, alpha=3):
    """Boucle GRASP."""
    best_path = None
    best_cost = float('inf')
    
    print(f"GRASP running (Alpha={alpha}, Iter={max_iterations})...")
    
    for i in range(max_iterations):
        start_node = random.randint(0, n - 1)
        # Phase 1
        sol = constructive_randomized_nearest_neighbor(n, matrix, alpha, start_node)
        # Phase 2
        sol_opt = local_search_2opt(sol, matrix)
        cost_opt = calculate_total_cost(sol_opt, matrix)
        
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_path = sol_opt
            
    return best_path, best_cost

if __name__ == "__main__":
    print("=== TEST Q5: GRASP ===")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    filename = os.path.join(base_dir, "data", "Input", "ali535.tsp")
    
    if os.path.exists(filename):
        n, mat = load_data(filename)
        path, cost = run_grasp(n, mat)
        print(f"Cout GRASP: {cost}")
        
        # Sauvegarde
        # On remonte de src/grasp/ (2 niveaux) vers Programmes/ puis vers data (donc ../../../data)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        out_file = os.path.join(base_dir, "data", "Solutions", "ali535_GRASP_solution.out")
        
        # S'assurer que le dossier existe
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        
        save_solution(out_file, path, cost)
