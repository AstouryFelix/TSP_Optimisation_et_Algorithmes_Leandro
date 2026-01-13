"""
Algorithme Constructif : Nearest Neighbor
==========================================
"""

import sys
import os

# Ajout du dossier parent au sys.path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.tsp_model import load_data, calculate_total_cost, save_solution, export_to_json

def constructive_nearest_neighbor(n, matrix, start_node=0):
    """Algorithme glouton du plus proche voisin."""
    unvisited = set(range(n))
    current_node = start_node
    path = [current_node]
    unvisited.remove(current_node)
    
    while unvisited:
        nearest_node = None
        min_dist = float('inf')
        
        for neighbor in unvisited:
            dist = matrix[current_node][neighbor]
            if dist < min_dist:
                min_dist = dist
                nearest_node = neighbor
        
        current_node = nearest_node
        path.append(current_node)
        unvisited.remove(current_node)
            
    return path

if __name__ == "__main__":
    print("=== TEST Q3: NEAREST NEIGHBOR ===")
    filename = "../../data/Input/100.in"
    if os.path.exists(filename):
        n, mat = load_data(filename)
        path = constructive_nearest_neighbor(n, mat)
        cost = calculate_total_cost(path, mat)
        print(f"Cout: {cost}")
