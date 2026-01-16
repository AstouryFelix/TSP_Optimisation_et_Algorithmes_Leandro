"""
Question 2 : Algorithme Exact (Branch & Bound)
==============================================
Ce module implémente une méthode exacte pour le TSP en utilisant le principe de Branch & Bound.
Algorithm based on: https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/
"""

import os
import math

from Tools.load_data      import *
from Tools.total_cost     import *
from Tools.export_to_json import *
from Tools.save_solution  import *
# Python3 program to solve 
# Traveling Salesman Problem using 
# Branch and Bound.
import math

class TSPBranchAndBound:
    def __init__(self, adjacency_matrix):
        self.adj = adjacency_matrix
        self.N = len(adjacency_matrix)
        self.final_res = float('inf')
        self.final_path = [None] * (self.N + 1)
        self.visited = [False] * self.N

    def copy_to_final(self, curr_path):
        """Stocke le meilleur chemin trouvé."""
        self.final_path[:self.N + 1] = curr_path[:]
        self.final_path[self.N] = curr_path[0]

    def first_min(self, i):
        """Trouve le coût minimum d'une arête connectée au sommet i."""
        min_val = float('inf')
        for k in range(self.N):
            if self.adj[i][k] < min_val and i != k:
                min_val = self.adj[i][k]
        return min_val

    def second_min(self, i):
        """Trouve le deuxième coût minimum d'une arête connectée au sommet i."""
        first, second = float('inf'), float('inf')
        for j in range(self.N):
            if i == j:
                continue
            if self.adj[i][j] <= first:
                second = first
                first = self.adj[i][j]
            elif self.adj[i][j] < second and self.adj[i][j] != first:
                second = self.adj[i][j]
        return second

    def tsp_rec(self, curr_bound, curr_weight, level, curr_path):
        """Fonction récursive principale (DFS avec élagage)."""
        
        # Cas de base : toutes les villes sont visitées
        if level == self.N:
            # Vérifier s'il y a un chemin de retour au début
            if self.adj[curr_path[level - 1]][curr_path[0]] != 0:
                curr_res = curr_weight + self.adj[curr_path[level - 1]][curr_path[0]]
                if curr_res < self.final_res:
                    self.copy_to_final(curr_path)
                    self.final_res = curr_res
            return

        # Explorer les villes suivantes
        for i in range(self.N):
            # Si la ville i n'est pas visitée et qu'il existe une arête
            if self.adj[curr_path[level - 1]][i] != 0 and not self.visited[i]:
                
                temp_bound = curr_bound
                curr_weight += self.adj[curr_path[level - 1]][i]

                # Calcul de la nouvelle borne inférieure (Lower Bound)
                if level == 1:
                    curr_bound -= ((self.first_min(curr_path[level - 1]) +
                                    self.first_min(i)) / 2)
                else:
                    curr_bound -= ((self.second_min(curr_path[level - 1]) +
                                    self.first_min(i)) / 2)

                # ÉLAGAGE (Pruning) : Si le coût estimé dépasse déjà le meilleur trouvé
                if curr_bound + curr_weight < self.final_res:
                    curr_path[level] = i
                    self.visited[i] = True
                    
                    # Appel Récursif
                    self.tsp_rec(curr_bound, curr_weight, level + 1, curr_path)
                    
                    # BACKTRACKING (Correction du bug ici)
                    # On annule simplement le marquage pour explorer d'autres branches
                    self.visited[i] = False 
                    self.visited[curr_path[level]] = False # Optionnel, par sécurité
                
                # Restauration des variables pour la boucle suivante (niveau actuel)
                curr_weight -= self.adj[curr_path[level - 1]][i]
                curr_bound = temp_bound

    def solve(self):
        """Initialise et lance l'algorithme."""
        curr_path = [-1] * (self.N + 1)
        curr_bound = 0

        # Calcul de la borne initiale
        for i in range(self.N):
            curr_bound += (self.first_min(i) + self.second_min(i))
        
        curr_bound = math.ceil(curr_bound / 2)

        # On commence au sommet 0
        self.visited[0] = True
        curr_path[0] = 0

        self.tsp_rec(curr_bound, 0, 1, curr_path)
        
        return self.final_res, self.final_path

# --- MAIN ---#
if __name__ == "__main__":
    file1 = "../data/Input/17.in"
    if os.path.exists(file1):
        print(f"\nTraitement de {file1}...")
        N, matrix = load_data(file1)

        solver = TSPBranchAndBound(matrix)
        cout, chemin = solver.solve()
        
        print("Coût Minimum :", cout)
        print("Chemin :", chemin)
        # 4. Sauvegarde
        base_name = os.path.basename(file1).replace(".in","").replace(".tsp","")
        save_solution(f"../data/Solutions/{base_name}_BB.out", chemin, cout)
        export_to_json(file1, matrix, chemin, cout, "_2opt")