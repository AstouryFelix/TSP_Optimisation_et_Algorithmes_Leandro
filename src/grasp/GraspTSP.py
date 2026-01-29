import random
import time
import sys, os

# Ajout du chemin racine pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.load_data      import load_data
from src.model.total_cost     import calculate_total_cost
from src.model.export_to_json import export_to_json
from src.model.save_solution  import save_solution
from src.local_search.LocalSearch import local_search_2opt

def constructive_randomized_nearest_neighbor(n, matrix, alpha=2, start_node=0):
    """
    Variante randomisée de l'heuristique constructive Nearest Neighbor.
    
    Principe : Au lieu de toujours choisir le voisin le plus proche,
    on construit une Liste Restreinte de Candidats (RCL) contenant
    les 'alpha' meilleurs voisins, puis on en choisit un au hasard.
    
    Paramètres:
    -----------
    n : int
        Nombre de villes
    matrix : list[list[int]]
        Matrice des distances
    alpha : int
        Taille de la RCL (1 = glouton pur, n = totalement aléatoire)
    start_node : int
        Ville de départ
    
    Retourne:
    ---------
    path : list[int]
        Chemin construit (liste des indices de villes)
    """
    unvisited = set(range(n))
    current_node = start_node
    path = [current_node]
    unvisited.remove(current_node)
    
    while unvisited:
        # Construire la liste de tous les candidats avec leurs distances
        candidates = []
        for neighbor in unvisited:
            dist = matrix[current_node][neighbor]
            candidates.append((dist, neighbor))
        
        # Trier par distance croissante
        candidates.sort(key=lambda x: x[0])
        
        # Construire la RCL (Restricted Candidate List)
        rcl_size = min(alpha, len(candidates))
        rcl = candidates[:rcl_size]
        
        # Choisir aléatoirement dans la RCL
        _, chosen_node = random.choice(rcl)
        
        # Avancer vers le noeud choisi
        current_node = chosen_node
        path.append(current_node)
        unvisited.remove(current_node)
        
    return path

def run_grasp(n, matrix, max_iterations=30, alpha=2, random_start=True, verbose=False):

    best_path = None
    best_cost = float('inf')
    history = []  # Pour tracer la convergence
    
    for i in range(max_iterations):
        # Diversification : choix du noeud de départ
        if random_start:
            start_node = random.randint(0, n - 1)
        else:
            start_node = 0
        
        # PHASE 1 : Construction randomisée
        solution = constructive_randomized_nearest_neighbor(n, matrix, alpha, start_node)
        
        # PHASE 2 : Amélioration locale (2-Opt)
        solution_improved = local_search_2opt(solution, matrix)
        cost_improved = calculate_total_cost(solution_improved, matrix)
        
        # Mise à jour de la meilleure solution
        if cost_improved < best_cost:
            best_cost = cost_improved
            best_path = solution_improved
            if verbose:
                print(f"  Iter {i+1:3d}: Nouvelle meilleure solution = {best_cost}")
        
        history.append(best_cost)
    
    return best_path, best_cost, history


if __name__ == "__main__":
    import sys
    import os

    ALPHA = 2
    ITERATIONS = 30
    
    # 1. Récupération du fichier (Confiance aveugle)
    if len(sys.argv) > 1:
        instance_file = sys.argv[1]
    else:
        # Fallback pour test simple
        instance_file = "instances/new_instances/random_5.in"
        print(f"Aucun fichier spécifié, utilisation par défaut : {instance_file}")

    # 2. Exécution directe
    if os.path.exists(instance_file):
        print(f"Resolution de {instance_file}")
        try:
            n, matrix = load_data(instance_file)
            
            t0 = time.time()
            path, cost, _ = run_grasp(n, matrix, max_iterations=ITERATIONS, alpha=ALPHA, verbose=True)
            elapsed = time.time() - t0
            
            print(f"Cout: {cost}")
            print(f"Temps: {elapsed:.2f}s")
            print(f"Params: alpha={ALPHA}, iter={ITERATIONS}")
            
            # 3. Sauvegarde dans Solutions/
            base_name = os.path.basename(instance_file).replace(".in", "").replace(".tsp", "")
            out_path = f"Solutions/{base_name}_grasp.out"
            
            os.makedirs("Solutions", exist_ok=True)
            
            save_solution(out_path, path, cost)
            export_to_json(instance_file, matrix, path, cost, "_grasp")
            
        except Exception as e:
            print(f"Erreur durant l'exécution : {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Erreur : Le fichier '{instance_file}' n'existe pas.")
        print(f"Usage: python {os.path.basename(__file__)} <fichier_instance>")