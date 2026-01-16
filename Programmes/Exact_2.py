"""
Question 2 : Algorithme Exact (Branch & Bound)
==============================================
Ce module implémente une méthode exacte pour le TSP en utilisant le principe de Branch & Bound.
Algorithm based on: https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/
"""

import sys
import os
import time
import math

# Ajout du path pour les modules src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from src.model.tsp_model import load_data, save_solution, export_to_json, calculate_total_cost
    from src.constructive.nearest_neighbor import constructive_nearest_neighbor
except ImportError:
    # Fallback pour compatibilité
    from Constructive_3 import load_data, save_solution, export_to_json, calculate_total_cost, constructive_nearest_neighbor

class BranchAndBoundTSP:
    def __init__(self, n, matrix):
        self.N = n
        self.adj = matrix
        
        self.final_res = float('inf')
        self.final_path = [None] * (n + 1)
        self.nodes_explored = 0

    def firstMin(self, i):
        """Find the minimum edge cost having an end at the vertex i"""
        mini = float('inf')
        for k in range(self.N):
            if self.adj[i][k] < mini and i != k:
                mini = self.adj[i][k]
        return mini

    def secondMin(self, i):
        """Find the second minimum edge cost having an end at the vertex i"""
        first, second = float('inf'), float('inf')
        for j in range(self.N):
            if i == j:
                continue
            if self.adj[i][j] <= first:
                second = first
                first = self.adj[i][j]
            elif self.adj[i][j] <= second and self.adj[i][j] != first:
                second = self.adj[i][j]
        return second

    def TSPRec(self, curr_bound, curr_weight, level, curr_path, visited):
        self.nodes_explored += 1
        
        # base case is when we have reached level N
        if level == self.N:
            # check if there is an edge from last vertex in path back to the first vertex
            if self.adj[curr_path[level - 1]][curr_path[0]] != 0:
                curr_res = curr_weight + self.adj[curr_path[level - 1]][curr_path[0]]
                if curr_res < self.final_res:
                    self.copyToFinal(curr_path)
                    self.final_res = curr_res
            return

        # for any other level iterate for all vertices
        for i in range(self.N):
            # Consider next vertex if it is not same (diagonal entry and not visited)
            if self.adj[curr_path[level-1]][i] != 0 and visited[i] == False:
                temp = curr_bound
                curr_weight += self.adj[curr_path[level - 1]][i]

                # different computation of curr_bound for level 2
                if level == 1:
                    curr_bound -= ((self.firstMin(curr_path[level - 1]) + self.firstMin(i)) / 2)
                else:
                    curr_bound -= ((self.secondMin(curr_path[level - 1]) + self.firstMin(i)) / 2)

                # curr_bound + curr_weight is the actual lower bound
                if curr_bound + curr_weight < self.final_res:
                    curr_path[level] = i
                    visited[i] = True
                    
                    self.TSPRec(curr_bound, curr_weight, level + 1, curr_path, visited)

                # Else prune or backtrack: reset changes
                curr_weight -= self.adj[curr_path[level - 1]][i]
                curr_bound = temp

                # Reset visited array part handled by recursion backtracking natively?
                # The prompt code manually resets visited array which looks complex/expensive inside python recursion loop
                # The prompt code logic for resetting 'visited' seems specific to their procedural style.
                # In standard recursion, we just set visited[i] = False after the call.
                visited[i] = False

    def copyToFinal(self, curr_path):
        self.final_path[:self.N + 1] = curr_path[:]
        self.final_path[self.N] = curr_path[0]

    def solve(self):
        # 1. Initialisation Heuristique (Optimisation perso)
        # On utilise le Nearest Neighbor pour avoir une bonne borne sup (final_res) dès le début.
        # Cela permet de 'prune' beaucoup de branches inutiles.
        try:
            nn_path = constructive_nearest_neighbor(self.N, self.adj)
            # nn_path de constructive est [0, 5, 2...] sans retour au 0 final explicite dans la liste généralement
            # On calcule son coût
            nn_cost = calculate_total_cost(nn_path, self.adj)
            self.final_res = nn_cost
            
            # On remplit final_path avec cette solution 'pauvre' mais valide au cas où
            for k in range(self.N):
                self.final_path[k] = nn_path[k]
            self.final_path[self.N] = nn_path[0]
            
            print(f"  [Init] Borne Supérieure (Nearest Neighbor) : {self.final_res}")
        except:
            print("  [Init] Impossible de lancer l'heuristique (Skip)")
        
        # 2. Préparation B&B
        curr_bound = 0
        curr_path = [-1] * (self.N + 1)
        visited = [False] * self.N

        # Compute initial bound
        for i in range(self.N):
            curr_bound += (self.firstMin(i) + self.secondMin(i))

        curr_bound = math.ceil(curr_bound / 2)

        # Start at vertex 0
        visited[0] = True
        curr_path[0] = 0

        # Call recursive
        self.TSPRec(curr_bound, 0, 1, curr_path, visited)
        
        # Format output
        # final_path contient le cycle complet (0 ... 0), on retourne juste la liste des villes
        result_path = self.final_path[:self.N]
        return result_path, self.final_res

# --- MAIN ---
if __name__ == "__main__":
    print("=== TEST QUESTION 2 : EXACT (BRANCH & BOUND) ===")
    
    # Chemins
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test sur petite instance
    test_file = os.path.join(base_dir, "test_5.in")
    if not os.path.exists(test_file):
        with open(test_file, "w") as f:
            f.write("5\n0 10 20 30 40\n10 0 15 25 35\n20 15 0 22 30\n30 25 22 0 18\n40 35 30 18 0")
            
    print(f"\nInstance Test : {test_file}")
    n, matrix = load_data(test_file)
    
    solver = BranchAndBoundTSP(n, matrix)
    t0 = time.time()
    path, cost = solver.solve()
    duration = time.time() - t0
    
    print(f"\nRésultat Exact : {cost}")
    print(f"Chemin : {path}")
    print(f"Noeuds explorés : {solver.nodes_explored}")
    print(f"Temps : {duration:.4f}s")
