"""
Question 5 : Métaheuristique GRASP
==================================
Implémentation de Greedy Randomized Adaptive Search Procedure.
Importe les modules précédents.
"""

from Constructive_3 import load_data, calculate_total_cost, save_solution, export_to_json
from LocalSearch_4 import local_search_2opt
import random
import time
import os

def constructive_randomized_nearest_neighbor(n, matrix, alpha=1, start_node=0):
    """
    Variante randomisée de l'heuristique constructive (Q3).
    Choisit un voisin au hasard parmi les 'alpha' meilleurs.
    """
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
        
        # RLC : Restricted Candidate List
        rcl_size = min(alpha, len(candidates))
        rcl = candidates[:rcl_size]
        
        chosen_dist, chosen_node = random.choice(rcl)
        
        current_node = chosen_node
        path.append(current_node)
        unvisited.remove(current_node)
        
    return path

def run_grasp(n, matrix, max_iterations=20, alpha=3):
    """Boucle principale GRASP."""
    best_path = None
    best_cost = float('inf')
    
    print(f"Run GRASP (Alpha={alpha}, Iter={max_iterations})...")
    
    for i in range(max_iterations):
        # Diversification : Début aléatoire
        start_node = random.randint(0, n - 1)
        
        # Phase 1
        sol = constructive_randomized_nearest_neighbor(n, matrix, alpha, start_node)
        
        # Phase 2
        sol_opt = local_search_2opt(sol, matrix)
        cost_opt = calculate_total_cost(sol_opt, matrix)
        
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_path = sol_opt
            # print(f"  > New Best @ Iter {i}: {best_cost}")
            
    return best_path, best_cost


# --- MAIN (TEST Q5) ---
if __name__ == "__main__":
    print("=== TEST QUESTION 5 : GRASP ===")
    
    # Utilisation de chemins absolus basés sur l'emplacement du script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # On remonte d'un cran (Programmes/ -> Root/)
    root_dir = os.path.dirname(base_dir)
    filename = os.path.join(root_dir, "data", "Input", "101.in")

    if not os.path.exists(filename):
        print(f"Fichier manquant : {filename}")
        exit()
        
    n, matrix = load_data(filename)
    
    # Paramètres fixés après expérimentation précédente
    best_alpha = 1
    iterations = 20
    
    t0 = time.time()
    path, cost = run_grasp(n, matrix, max_iterations=iterations, alpha=best_alpha)
    
    print(f"\nRésultat Final GRASP (Alpha={best_alpha}) : {cost}")
    print(f"Temps total : {(time.time()-t0):.2f}s")
    
    # Correction chemin de sortie
    out_file = os.path.join(root_dir, "data", "Solutions", "101_GRASP_Final.out")
    save_solution(out_file, path, cost)
    # Note: export_to_json pourrait aussi avoir besoin d'une correction de path si elle n'est pas robuste
    # Mais on utilise Generate_Visu_Data désormais.
    # export_to_json(filename, matrix, path, cost, "_GRASP_Final") # Cette ligne est redondante ou problématique si mal configurée.
