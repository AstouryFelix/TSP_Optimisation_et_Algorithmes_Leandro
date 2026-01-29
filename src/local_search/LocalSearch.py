import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.load_data      import *
from src.model.total_cost     import *
from src.model.export_to_json import *
from src.model.save_solution  import *
from src.constructive.Constructive import constructive_nearest_neighbor
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
    import sys
    import os
    import time
    
    print("=== LOCAL SEARCH (2-OPT) ===")
    
    # 1. Récupération du fichier
    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
    else:
        # Test pas défaut
        instance_file = "instances/new_instances/random_5.in"
        print(f"Aucun fichier spécifié, utilisation par défaut : {instance_file}")
        
    # 2. Exécution directe
    if os.path.exists(instance_file):
        print(f"Resolving: {instance_file}")
        try:
            # 1. Chargement
            n, matrix = load_data(instance_file)
            
            # 2. Construction Initiale (Q3)
            t0 = time.time()
            init_path = constructive_nearest_neighbor(n, matrix)
            init_cost = calculate_total_cost(init_path, matrix)
            print(f"Initial (NN) : {init_cost} ({(time.time()-t0):.2f}s)")
            
            # 3. Amélioration (Q4)
            t1 = time.time()
            opt_path = local_search_2opt(init_path, matrix)
            opt_cost = calculate_total_cost(opt_path, matrix)
            elapsed = time.time()-t1
            print(f"Optimisé (2-Opt) : {opt_cost} ({elapsed:.2f}s)")
            print(f"Gain : {init_cost - opt_cost}")
            
            # 4. Sauvegarde
            base_name = os.path.basename(instance_file).replace(".in","").replace(".tsp","")
            outfile = f"Solutions/{base_name}_local_search.out"
            
            os.makedirs("Solutions", exist_ok=True)
            
            save_solution(outfile, opt_path, opt_cost)
            export_to_json(instance_file, matrix, opt_path, opt_cost, "_local_search")
            print("Sortie générée dans Solutions/")
            
        except Exception as e:
            print(f"Erreur: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print(f"Erreur : Le fichier '{instance_file}' n'existe pas.")
        print(f"Usage: python {os.path.basename(__file__)} <fichier_instance>")
