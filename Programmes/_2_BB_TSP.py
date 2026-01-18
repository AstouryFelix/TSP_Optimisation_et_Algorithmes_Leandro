import os
import heapq
import numpy as np
from scipy.optimize import linear_sum_assignment

from Tools.load_data      import *
from Tools.total_cost     import *
from Tools.export_to_json import *
from Tools.save_solution  import *

class TSP_ILP_Solver:
    def __init__(self, cost_matrix):
        self.original_matrix = np.array(cost_matrix, dtype=float)
        self.N = len(cost_matrix)
        self.best_cost = float('inf')
        self.best_path = []
        
        # Pre-fill diagonal with infinity to prevent self-loops in relaxation
        np.fill_diagonal(self.original_matrix, float('inf'))

    def solve_relaxation(self, matrix):
        """
        Solves the Linear Assignment Problem (The 'Simplex' Step).
        Relaxation: x_ij can be 0 or 1, constraints: 1 entry, 1 exit per city.
        Ignores subtours (connectivity).
        """
        # Scipy's linear_sum_assignment is the efficient solver for this relaxation
        row_ind, col_ind = linear_sum_assignment(matrix)
        
        # Calculate the Lower Bound (Z_L*)
        cost = matrix[row_ind, col_ind].sum()
        
        # Reconstruct the edges chosen by the solver
        edges = []
        for r, c in zip(row_ind, col_ind):
            edges.append((r, c))
            
        return cost, edges

    def find_subtours(self, edges):
        """
        Checks if the relaxation result forms a single valid tour
        or multiple disconnected loops (subtours).
        """
        adj = {u: v for u, v in edges}
        visited = set()
        subtours = []
        
        for i in range(self.N):
            if i in visited:
                continue
            
            # Trace the cycle
            curr = i
            cycle = []
            while curr not in visited:
                visited.add(curr)
                cycle.append(curr)
                if curr in adj:
                    curr = adj[curr]
                else:
                    break # Should not happen in Assignment Problem
            
            subtours.append(cycle)
            
        return subtours

    def solve(self, verbose=True):
        # Priority Queue for Best-First Search
        # Format: (LowerBound, unique_id, current_matrix, forbidden_edges_count)
        pq = []
        node_id = 0
        last_printed_lb = -1.0 
        last_print_pr = -1.0

        # --- ROOT NODE ---
        # Solve initial P_L (Relaxation) [cite: 13, 142]
        lb, edges = self.solve_relaxation(self.original_matrix)
        heapq.heappush(pq, (lb, node_id, self.original_matrix.copy(), []))
        
        iter_count = 0
        if verbose:
            print(f"{'ITER':<6} {'BOUND':<10} {'BEST_REF':<10} {'ACTION'}")
            print("-" * 50)

        while pq:
            iter_count += 1
            # 1. Selection: Pick node with best Lower Bound [cite: 232]
            curr_lb, _, curr_matrix, debug_path = heapq.heappop(pq)
            
            # 2. Pruning (Elagage): If LB > Best known solution, stop [cite: 236]
            if curr_lb >= self.best_cost and curr_lb > last_print_pr :
                if verbose:
                    print(f"{iter_count:<6} {curr_lb:<10.2f} {self.best_cost:<10.2f} ✂️ Pruned")
                    last_print_pr = curr_lb
                continue

            # 3. Analyze Relaxation Result
            # We re-solve purely to get the edges (cost is already in curr_lb)
            # In optimized C++ we wouldn't re-solve, but for Python clarity we do.
            _, current_edges = self.solve_relaxation(curr_matrix)
            subtours = self.find_subtours(current_edges)

            # 4. Check for Integer/Valid Solution [cite: 166]
            # If only 1 subtour containing all N cities -> Valid TSP Tour
            if len(subtours) == 1 and len(subtours[0]) == self.N:
                if curr_lb < self.best_cost:
                    self.best_cost = curr_lb
                    # Convert cycle [0, 2, 1] to edges for storage if needed
                    self.best_path = subtours[0] + [subtours[0][0]]
                    if verbose:
                        print(f"{iter_count:<6} {curr_lb:<10.2f} {self.best_cost:<10.2f} 🏆 SOLUTION FOUND")
                continue

            # 5. Branching (Separation) [cite: 45, 173]
            # We found subtours (e.g., 0-1-0). This is invalid.
            # We must break the subtour.
            # Strategy: Pick the smallest subtour and branch on its edges.
            # For a subtour i->j->k->i, we create branches forbidding (i,j), then (j,k)...
            
            shortest_subtour = min(subtours, key=len)

            if verbose and curr_lb > last_printed_lb:
                print(f"{iter_count:<6} {curr_lb:<10.2f} {self.best_cost:<10.2f} 🌱 Branching on subtour len {len(shortest_subtour)}")
                last_printed_lb = curr_lb

            # Convert cities list to edges: [0, 1, 2] -> [(0,1), (1,2), (2,0)]
            edges_to_break = []
            for k in range(len(shortest_subtour)):
                u, v = shortest_subtour[k], shortest_subtour[(k+1)%len(shortest_subtour)]
                edges_to_break.append((u, v))

            # Create a child node for each edge removal
            # Branch i: Forbid edge edges_to_break[i]
            for u, v in edges_to_break:
                child_matrix = curr_matrix.copy()
                # Constraint: x_uv = 0 (set cost to infinity)
                child_matrix[u, v] = float('inf')
                
                # Solve relaxation for child to get new LB [cite: 162]
                child_lb, _ = self.solve_relaxation(child_matrix)
                
                # Only add if potential solution is better than best known
                if child_lb < self.best_cost:
                    node_id += 1
                    heapq.heappush(pq, (child_lb, node_id, child_matrix, []))

        return self.best_cost, self.best_path

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Standard testing matrix
    inf = float('inf')
    matrix = [
        [inf, 10,  15,  20],
        [10,  inf, 35,  25],
        [15,  35,  inf, 30],
        [20,  25,  30,  inf]
    ]

    print("\n--- Solving with Assignment Problem Relaxation (Simplex-like) ---")
    solver = TSP_ILP_Solver(matrix)
    cost, path = solver.solve()
    
    print("\nFinal Result:")
    print(f"Minimum Cost: {cost}")
    print(f"Path: {path}")

if __name__ == "__main__":
    # Example matrix (Cost to go from i to j)
    # Inf on diagonal
    inf = float('inf')
    file1 = "../data/Input/17.in"
    N, matrix = load_data(file1)
    print("launched")
    solver = TSP_ILP_Solver(matrix)
    cost, path = solver.solve()
    print("Optimization finished.")
    print(f"Min Cost: {cost}")
    print(f"Path: {path}")
    base_name = os.path.basename(file1).replace(".in","").replace(".tsp","")
    save_solution(f"../data/Solutions/{base_name}_BB.out", path, cost)
    export_to_json(file1, matrix, path, cost, "_BB")