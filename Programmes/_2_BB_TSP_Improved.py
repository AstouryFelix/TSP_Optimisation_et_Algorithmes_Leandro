import os
import heapq
import numpy as np
from scipy.optimize import linear_sum_assignment

from Tools.load_data      import *
from Tools.total_cost     import *
from Tools.save_solution  import *
from Tools.export_to_json import *


class TSP_ILP_Solver:
    def __init__(self, cost_matrix):
        self.matrix    = np.array(cost_matrix, dtype=float)
        self.N         = len(cost_matrix)
        self.best_cost = float('inf')
        self.best_path = []
        
        # Pre-fill diagonal
        np.fill_diagonal(self.matrix, float('inf'))

    def solve_relaxation(self):
        """
        Solves the Assignment Problem on the CURRENT state of self.matrix
        """
        # Résout le problème manière optimale, sans vérifier la faisabilité du TSP
        row_ind, col_ind = linear_sum_assignment(self.matrix)
        cost = self.matrix[row_ind, col_ind].sum()
        
        # Reconstruct edges
        edges = []
        for r, c in zip(row_ind, col_ind):
            edges.append((r, c))
        
        return cost, edges

    def find_subtours(self, edges):
        """ Converts edge list to a list of cycles (subtours) """
        adj = {u: v for u, v in edges}
        visited = set()
        subtours = []
        
        for i in range(self.N):
            if i in visited:
                continue
            curr = i
            cycle = []
            while curr not in visited:
                visited.add(curr)
                cycle.append(curr)
                curr = adj[curr]
            subtours.append(cycle)
        return subtours

    def solve(self):

        print(f"{'DEPTH':<6} {'BOUND':<10} {'BEST_REF':<10} {'ACTION'}")
        print("-" * 50)
        
        # Initial Relaxation
        initial_lb, initial_edges = self.solve_relaxation()
        
        # Start the recursive Branch and Bound
        self._recursive_solve(initial_lb, initial_edges, depth=0)
        
        return self.best_cost, self.best_path

    def _recursive_solve(self, current_lb, current_edges, depth):
        if current_lb >= self.best_cost:
            return

        subtours = self.find_subtours(current_edges)

        # Une solution valide n'a qu'un circuit de longueur N
        if len(subtours) == 1 and len(subtours[0]) == self.N:
            if current_lb < self.best_cost:
                self.best_cost = current_lb
                self.best_path = subtours[0] + [subtours[0][0]] 
                print(f"{depth:<6} {current_lb:<10.2f} {self.best_cost:<10.2f} 🏆 NEW OPTIMUM")
            return

        # 4. Branching Strategy
        # Pick smallest subtour to break (heuristic)
        shortest_subtour = min(subtours, key=len)
        
        # Identify edges to branch on: i -> j
        edges_to_break = []
        for k in range(len(shortest_subtour)):
            u = shortest_subtour[k]
            v = shortest_subtour[(k+1) % len(shortest_subtour)]
            edges_to_break.append((u, v))

        # 5. Look-ahead (Strong Branching / Sorting)
        # We calculate the LB for all children BEFORE recursing.
        # This allows us to visit the most promising child first (DFS Best-First).
        candidates = []

        for u, v in edges_to_break:
            # --- MODIFY STATE ---
            old_val = self.matrix[u, v]
            if old_val == float('inf'): continue # Already forbidden

            self.matrix[u, v] = float('inf') # Forbid edge
            
            # Solve relaxation
            child_lb, child_edges = self.solve_relaxation()
            
            # --- BACKTRACK STATE (Immediately) ---
            self.matrix[u, v] = old_val 
            
            # If feasible and promising, add to candidates
            if child_lb < self.best_cost:
                candidates.append((child_lb, u, v, child_edges))

        # Sort candidates by Lower Bound (Greedy / Best-First)
        candidates.sort(key=lambda x: x[0])

        # 6. Recurse into sorted candidates
        for child_lb, u, v, child_edges in candidates:
            # --- MODIFY STATE PERMANENTLY FOR RECURSION ---
            old_val = self.matrix[u, v]
            self.matrix[u, v] = float('inf')

            # Recurse
            self._recursive_solve(child_lb, child_edges, depth + 1)

            # --- BACKTRACK (Restore for next sibling) ---
            self.matrix[u, v] = old_val

if __name__ == "__main__":
    inf = float('inf')
    file1 = "../data/Input/17.in"
    N, matrix = load_data(file1)
    solver = TSP_ILP_Solver(matrix)
    cost, path = solver.solve()
    print("Optimization finished.")
    print(f"Min Cost: {cost}")
    print(f"Path: {path}")
    base_name = os.path.basename(file1).replace(".in","").replace(".tsp","")
    save_solution(f"../data/Solutions/{base_name}_BB.out", path, cost)
    export_to_json(file1, matrix, path, cost, "_BB")