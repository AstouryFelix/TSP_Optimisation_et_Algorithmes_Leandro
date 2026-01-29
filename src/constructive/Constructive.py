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
    import sys
    import os
    import time

    # 1. Récupération du fichier (Confiance aveugle)
    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
    else:
        # Fallback pour test simple
        instance_file = "instances/new_instances/random_5.in"
        print(f"Aucun fichier spécifié, utilisation par défaut : {instance_file}")

    # 2. Exécution directe
    if os.path.exists(instance_file):
        print(f"=== NEAREST NEIGHBOR ===")
        print(f"Resolving: {instance_file}")
        
        try:
            n, mat = load_data(instance_file)
            
            t0 = time.time()
            
            path = constructive_nearest_neighbor(n, mat)
            
            elapsed = time.time() - t0
            cost = calculate_total_cost(path, mat)
            
            print(f"Cout NN: {cost}")
            print(f"Temps: {elapsed:.4f}s")
            
            # Sauvegarde
            base_name = os.path.basename(instance_file).replace(".in","").replace(".tsp","")
            outfile = f"Solutions/{base_name}_constructive.out"
            
            os.makedirs("Solutions", exist_ok=True)
            
            save_solution(outfile, path, cost)
            export_to_json(instance_file, mat, path, cost, "_constructive")
            print("Sortie générée dans Solutions/")
            
        except Exception as e:
            print(f"Erreur: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Erreur : Le fichier '{instance_file}' n'existe pas.")
        print(f"Usage: python {os.path.basename(__file__)} <fichier_instance>")