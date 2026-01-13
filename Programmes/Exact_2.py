"""
Question 2 : Algorithme Exact (Branch & Bound)
==============================================
Ce module implémente une méthode exacte pour le TSP.
ATTENTION: Complexité Exponentielle ! Ne pas utiliser pour n > 20.
"""

from Constructive_3 import load_data, save_solution, export_to_json, calculate_total_cost
import time
import os
import math

class BranchAndBoundTSP:
    def __init__(self, n, matrix):
        self.n = n
        self.matrix = matrix
        self.best_cost = float('inf')
        self.best_path = []
        self.nodes_explored = 0
        
        # Pré-calcul des 2 arêtes les plus courtes pour chaque nœud (pour la borne inférieure)
        self.min_edges = []
        for i in range(n):
            row = sorted([matrix[i][j] for j in range(n) if i != j])
            if len(row) >= 2:
                self.min_edges.append(row[:2])
            else:
                self.min_edges.append([row[0], float('inf')]) # Cas dégénéré

    def lower_bound(self, path, visited_mask):
        """
        Calcule une borne inférieure simple (Lower Bound).
        LB = (Coût actuel + Estimation du reste)
        Estimation = Somme(min_edges)/2 pour les nœuds non connectés.
        """
        # 1. Coût du chemin partiel
        current_cost = 0
        last_node = path[-1]
        for i in range(len(path)-1):
            current_cost += self.matrix[path[i]][path[i+1]]
            
        # 2. Estimation pour clore le cycle
        # Chaque nœud doit avoir degré 2.
        # - Nœuds internes du path : degré 2 déjà fixé (sauf start et end)
        # - Start node : a 1 arête fixée (vers le 2e), a besoin d'1 autre (retour)
        # - End node : a 1 arête fixée (venant du précédent), a besoin d'1 autre (vers nouveau)
        # - Unvisited nodes : ont besoin de 2 arêtes
        
        # Simplification pour B&B rapide :
        # On regarde le coût minimal pour sortir/entrer des nœuds libres.
        estimated_remaining = 0
        
        # Pour le dernier nœud visité, il doit aller vers un non-visité (min sortant)
        # Pour le start node, il doit recevoir du dernier (min entrant)
        # Pour les non-visités, ils doivent avoir 2 arêtes.
        
        # Implémentation simplifiée : coût actuel + borne simple
        # Si on veut être vraiment efficace, il faudrait l'algo de Held-Karp (1-Tree).
        # Ici on va faire un DFS simple avec pruning sur le coût courant.
        # LB = current_cost. C'est faible mais correct.
        
        return current_cost

    def _dfs(self, current_node, current_cost, path, visited_mask):
        self.nodes_explored += 1
        
        # Pruning (Si coût actuel dépasse déjà le meilleur connu)
        if current_cost >= self.best_cost:
            return

        # Si tous visités, on boucle sur le début
        if len(path) == self.n:
            total_cost = current_cost + self.matrix[current_node][path[0]]
            if total_cost < self.best_cost:
                self.best_cost = total_cost
                self.best_path = list(path)
                print(f"  > Nouvelle meilleure solution : {self.best_cost}")
            return

        # Exploration des voisins non visités
        # Heuristique : Trier les voisins par distance pour trouver une bonne solution vite
        candidates = []
        for neighbor in range(self.n):
            if not (visited_mask & (1 << neighbor)):
                candidates.append((neighbor, self.matrix[current_node][neighbor]))
        
        candidates.sort(key=lambda x: x[1])
        
        for neighbor, dist in candidates:
             self._dfs(neighbor, current_cost + dist, path + [neighbor], visited_mask | (1 << neighbor))

    def solve(self):
        # Initialisation avec une heuristique (Nearest Neighbor) pour avoir une borne sup initiale
        # Cela permet de couper beaucoup plus vite !
        import Constructive_3
        nn_path = Constructive_3.constructive_nearest_neighbor(self.n, self.matrix)
        nn_cost = Constructive_3.calculate_total_cost(nn_path, self.matrix)
        
        self.best_cost = nn_cost
        self.best_path = nn_path
        print(f"  [Init] Solution Heuristique (Born Sup) : {self.best_cost}")
        
        # Lancement DFS (Node 0 fixé)
        self._dfs(0, 0, [0], 1) # Mask 1 = Node 0 visité
        
        return self.best_path, self.best_cost

# --- MAIN (QUESTION 2) ---
if __name__ == "__main__":
    print("=== TEST QUESTION 2 : EXACT (BRANCH & BOUND) ===")
    
    # Test sur petite instance créée
    filename = "test_5.in"
    if not os.path.exists(filename):
        # Création à la volée si existe pas
        with open(filename, "w") as f:
            f.write("5\n0 10 20 30 40\n10 0 15 25 35\n20 15 0 22 30\n30 25 22 0 18\n40 35 30 18 0")
            
    print(f"\nInstance Test : {filename}")
    n, matrix = load_data(filename)
    
    solver = BranchAndBoundTSP(n, matrix)
    t0 = time.time()
    path, cost = solver.solve()
    duration = time.time() - t0
    
    print(f"\nRésultat Exact : {cost}")
    print(f"Chemin : {path}")
    print(f"Noeuds explorés : {solver.nodes_explored}")
    print(f"Temps : {duration:.4f}s")
    
    # Test sur plus gros si possible (Attention au temps !)
    filename_real = "../data/Input/100.in"
    if os.path.exists(filename_real):
        print(f"\n--- Test sur instance moyenne ({filename_real}) ---")
        print("ATTENTION : On va lancer mais avec un timeout simulé (ou manuel) car n=100 est impossible en exact.")
        print("On ne le fait pas ici pour ne pas bloquer l'exécution.")
