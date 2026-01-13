"""
Question 4 : Recherche Locale (2-Opt)
=====================================
Ce module implémente l'amélioration locale 2-Opt.
Il importe les outils nécessaires depuis Constructive_3.py.
"""

from Constructive_3 import load_data, calculate_total_cost, constructive_nearest_neighbor, save_solution, export_to_json
import time
import os

def local_search_2opt(path, matrix):
    """
    Améliore un chemin existant en utilisant l'opérateur "2-opt" (First Improvement).
    """
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
                    # Inversion du segment
                    best_path[i+1 : j+1] = best_path[i+1 : j+1][::-1]
                    improved = True
    return best_path


# --- MAIN (TEST Q4) ---
if __name__ == "__main__":
    print("=== TEST QUESTION 4 : LOCAL SEARCH (2-OPT) ===")
    
    # Liste des fichiers à tester
    files = ["../data/Input/100.in", "../data/Input/ali535.tsp"]
    
    for filename in files:
        if not os.path.exists(filename):
            print(f"Fichier manquant: {filename}")
            continue
            
        print(f"\n--- Instance : {os.path.basename(filename)} ---")
        
        # 1. Chargement
        n, matrix = load_data(filename)
        
        # 2. Construction Initiale (Q3)
        t0 = time.time()
        init_path = constructive_nearest_neighbor(n, matrix)
        init_cost = calculate_total_cost(init_path, matrix)
        print(f"Initial (NN) : {init_cost} ({(time.time()-t0):.2f}s)")
        
        # 3. Amélioration (Q4)
        t1 = time.time()
        opt_path = local_search_2opt(init_path, matrix)
        opt_cost = calculate_total_cost(opt_path, matrix)
        print(f"Optimisé (2-Opt) : {opt_cost} ({(time.time()-t1):.2f}s)")
        print(f"Gain : {init_cost - opt_cost}")
        
        # 4. Sauvegarde
        base_name = os.path.basename(filename).replace(".in","").replace(".tsp","")
        save_solution(f"../data/Solutions/{base_name}_2opt.out", opt_path, opt_cost)
        export_to_json(filename, matrix, opt_path, opt_cost, "_2opt")
