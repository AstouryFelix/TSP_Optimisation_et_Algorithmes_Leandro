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
        # Il nous faut un arbre à naviguer
        # Il doit avoir une valeur, une solution, et la liste des contraintes lui étant attriubés

    def _dfs(self, current_node, current_cost, path, visited_mask):
        # Jsp si c'est utile
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
        # Refaire entièrement
        # Le speudocode est dans le word
        # Il faut trouver un solver pour résoudre le simplex.
        pass

# --- MAIN (QUESTION 2) ---
if __name__ == "__main__":
    print("=== TEST QUESTION 2 : EXACT (BRANCH & BOUND) ===")
    
    # Test sur petite instance créée
    filename = "../data/Input/17.in"
    if os.path.exists(filename):  
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
    filename = "../data/Input/100.in"
    if os.path.exists(filename):
        n, matrix = load_data(filename)
        solver = BranchAndBoundTSP(n, matrix)
        t0 = time.time()
        path, cost = solver.solve()
        duration = time.time() - t0

        print(f"\nRésultat Exact : {cost}")
        print(f"Chemin : {path}")
        print(f"Noeuds explorés : {solver.nodes_explored}")
        print(f"Temps : {duration:.4f}s")