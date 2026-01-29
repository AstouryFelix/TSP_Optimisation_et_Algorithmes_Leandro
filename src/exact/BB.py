import numpy as np
from scipy.optimize import linprog
import heapq
import os
import json

from src.model.load_data      import *
from src.model.total_cost     import *
from src.model.export_to_json import *
from src.model.save_solution  import *

class TSP_ILP_Solver:
    def __init__(self, matrix):
        self.matrix = np.array(matrix)
        self.n = len(matrix)
        self.best_cost = float('inf')
        self.best_path = []
        self.nodes_explored = 0
        
        self.x_map = {}
        self.idx_to_ij = {}
        count = 0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.x_map[(i, j)] = count
                    self.idx_to_ij[count] = (i, j)
                    count += 1
        self.num_x = count
        
        self.u_start_idx = self.num_x
        self.num_u = self.n - 1 
        self.total_vars = self.num_x + self.num_u
        
        self.c = np.zeros(self.total_vars)
        for (i, j), idx in self.x_map.items():
            self.c[idx] = self.matrix[i][j]

        self.A_eq = []
        self.b_eq = []
        
        for i in range(self.n):
            row = np.zeros(self.total_vars)
            for j in range(self.n):
                if i != j:
                    row[self.x_map[(i, j)]] = 1
            self.A_eq.append(row)
            self.b_eq.append(1)
            
        for j in range(self.n):
            row = np.zeros(self.total_vars)
            for i in range(self.n):
                if i != j:
                    row[self.x_map[(i, j)]] = 1
            self.A_eq.append(row)
            self.b_eq.append(1)

        self.A_ub = []
        self.b_ub = []
        
        for i in range(1, self.n):
            for j in range(1, self.n):
                if i != j:
                    row = np.zeros(self.total_vars)
                    row[self.u_start_idx + (i - 1)] = 1 
                    row[self.u_start_idx + (j - 1)] = -1
                    if (i, j) in self.x_map:
                         row[self.x_map[(i, j)]] = self.n - 1
                    
                    self.A_ub.append(row)
                    self.b_ub.append(self.n - 2)

    def solve_relaxation(self, fixed_vars):
        """
        Solves the LP relaxation with current branching constraints.
        fixed_vars: dict {var_index: value (0 or 1)}
        """
        
        bounds = []
        for k in range(self.num_x):
            if k in fixed_vars:
                val = fixed_vars[k]
                bounds.append((val, val))
            else:
                bounds.append((0, 1)) 
        
        for _ in range(self.num_u):
            bounds.append((2, self.n)) 

        res = linprog(c=self.c, A_eq=self.A_eq, b_eq=self.b_eq, 
                      A_ub=self.A_ub, b_ub=self.b_ub, bounds=bounds, method='highs')
        
        return res

    def extract_path(self, x_values):
        """Reconstructs the tour from binary x variables."""
        adj = {}
        for idx, val in enumerate(x_values[:self.num_x]):
            if val > 0.9: 
                i, j = self.idx_to_ij[idx]
                adj[i] = j
        
        path = [0]
        current = 0
        visited = {0}
        while len(path) < self.n:
            if current not in adj:
                return None # Broken path
            next_node = adj[current]
            if next_node in visited:
                return None # Subtour
            visited.add(next_node)
            path.append(next_node)
            current = next_node
        # Close loop check
        if adj[current] != 0: 
            return None
        return [p + 1 for p in path] # Return 1-based indexing for output

    def solve(self, verbose=True):
        # Priority Queue for Best-First Search strategy (minimizing lower bound)
        # Item: (lower_bound, unique_id, fixed_vars_dict)
        pq = []
        heapq.heappush(pq, (0, 0, {}))
        node_counter = 0

        if verbose:
            print(f"Starting Branch and Bound for N={self.n}...")
        
        while pq:
            lb, _, fixed_vars = heapq.heappop(pq)
            
            # Pruning: If potential LB is worse than best found, discard 
            if lb >= self.best_cost and self.best_cost != float('inf'):
                continue

            # Solve Relaxation [cite: 1, 8]
            res = self.solve_relaxation(fixed_vars)

            # If infeasible, prune [cite: 234]
            if not res.success:
                continue
            
            current_cost = res.fun
            
            # Pruning again with tighter bound from actual calculation
            if current_cost >= self.best_cost:
                continue

            # Check integrality
            x_vals = res.x[:self.num_x]
            is_integer = True
            fractional_var_idx = -1
            closest_dist = 0.5
            
            for idx, val in enumerate(x_vals):
                dist = abs(val - 0.5)
                if dist < 0.499: # It is fractional (close to 0.5)
                    is_integer = False
                    # Heuristic: Branch on variable closest to 0.5 (most ambiguous)
                    if dist < closest_dist:
                        closest_dist = dist
                        fractional_var_idx = idx
            
            if is_integer:
                # We found a valid integer solution!
                # Since we explored by best bound, this might be optimal, 
                # but we must continue until queue empty or bounds exceed this.
                path = self.extract_path(res.x)
                if path and current_cost < self.best_cost:
                    if verbose:
                        print(f"New best integer solution found: Cost {current_cost}")
                    self.best_cost = current_cost
                    self.best_path = path
            else:
                # Branching [cite: 228, 230]
                # Create two children: x_ij = 0 and x_ij = 1
                
                # Child 1: x_k = 0
                vars_0 = fixed_vars.copy()
                vars_0[fractional_var_idx] = 0
                node_counter += 1
                heapq.heappush(pq, (current_cost, node_counter, vars_0))
                
                # Child 2: x_k = 1
                vars_1 = fixed_vars.copy()
                vars_1[fractional_var_idx] = 1
                node_counter += 1
                heapq.heappush(pq, (current_cost, node_counter, vars_1))
                
            self.nodes_explored += 1
            if self.nodes_explored % 100 == 0 and verbose:
                print(f"Nodes explored: {self.nodes_explored}, Queue size: {len(pq)}, Current Best: {self.best_cost}")

        return self.best_cost, self.best_path

if __name__ == "__main__":
    import os
    inf = float('inf')
    
    # Recherche du fichier 17.in
    possible_paths = [
        "instances/exact/17.in",
        "../instances/exact/17.in",
        "../../instances/exact/17.in"
    ]
    
    file1 = None
    for p in possible_paths:
        if os.path.exists(p):
            file1 = p
            break
            
    if not file1:
        print("Erreur : fichier 17.in introuvable dans instances/exact/")
        import sys
        sys.exit(1)

    print(f"Chargement de {file1}...")
    N, matrix = load_data(file1)
    solver = TSP_ILP_Solver(matrix)
    cost, path = solver.solve()
    print("Optimization finished.")
    print(f"Min Cost: {cost}")
    print(f"Path: {path}")
    
    base_name = os.path.basename(file1).replace(".in","").replace(".tsp","")
    outfile = f"Solutions/{base_name}_exact.out"
    save_solution(outfile, path, cost, zero_based=False)
    export_to_json(file1, matrix, path, cost, "_BB")