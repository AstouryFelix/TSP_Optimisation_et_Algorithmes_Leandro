"""
Algorithme Exact : Branch & Bound
=================================
"""

import sys
import os

# Ajout du dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.tsp_model import load_data, calculate_total_cost
from src.constructive.nearest_neighbor import constructive_nearest_neighbor

class BranchAndBoundTSP:
    def __init__(self, n, matrix):
        self.n = n
        self.matrix = matrix
        self.best_cost = float('inf')
        self.best_path = []
        self.nodes_explored = 0
        
    def _dfs(self, current_node, current_cost, path, visited_mask):
        self.nodes_explored += 1
        if current_cost >= self.best_cost: return

        if len(path) == self.n:
            total_cost = current_cost + self.matrix[current_node][path[0]]
            if total_cost < self.best_cost:
                self.best_cost = total_cost
                self.best_path = list(path)
                print(f"  > New Best: {self.best_cost}")
            return

        candidates = []
        for neighbor in range(self.n):
            if not (visited_mask & (1 << neighbor)):
                candidates.append((neighbor, self.matrix[current_node][neighbor]))
        
        candidates.sort(key=lambda x: x[1])
        
        for neighbor, dist in candidates:
             self._dfs(neighbor, current_cost + dist, path + [neighbor], visited_mask | (1 << neighbor))

    def solve(self):
        # Init avec NN pour Borne Sup
        nn_path = constructive_nearest_neighbor(self.n, self.matrix)
        self.best_cost = calculate_total_cost(nn_path, self.matrix)
        self.best_path = nn_path
        print(f"  [Init] Upper Bound (NN): {self.best_cost}")
        
        self._dfs(0, 0, [0], 1)
        return self.best_path, self.best_cost

if __name__ == "__main__":
    print("=== TEST Q2: EXACT ===")
    # Pour tester, on crée une matrice random à la volée ou petit fichier
    # Ici, juste un print pour valider la structure
    print("Pret a l'emploi.")
