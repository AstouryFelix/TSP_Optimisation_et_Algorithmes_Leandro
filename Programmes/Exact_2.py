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

from Tools.load_data      import *
from Tools.total_cost     import *
from Tools.export_to_json import *
from Tools.save_solution  import *

class BranchAndBoundTSP:
    def __init__(self, n, matrix):
        self.N = n
        self.adj = matrix
        
        self.final_res = float('inf')
        self.final_path = [None] * (n + 1)
        self.nodes_explored = 0
        # Il nous faut un arbre à naviguer
        # Il doit avoir une valeur, une solution, et la liste des contraintes lui étant attriubés

    def _dfs(self, current_node, current_cost, path, visited_mask):
        # Jsp si c'est utile
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
        # Refaire entièrement
        # Le speudocode est dans le word
        # Il faut trouver un solver pour résoudre le simplex.
        pass

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
