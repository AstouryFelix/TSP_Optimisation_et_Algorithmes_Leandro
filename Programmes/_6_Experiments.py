"""
Question 6 : Expérimentations et Analyse
========================================
Ce script compare les performances (Temps et Qualité) de tous les algorithmes implémentés.
Il génère des instances aléatoires de taille N et lance les solveurs.

Algorithmes comparés :
1. Exact (Branch & Bound) - Limité à N <= 12 pour le temps
2. Constructive (Nearest Neighbor)
3. Local Search (2-Opt)
4. GRASP
"""

import time
import random
import math
import sys
import os

# Import des algo (Nouvelle Structure src)
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model.tsp_model import build_distance_matrix, calculate_total_cost
from src.constructive.nearest_neighbor import constructive_nearest_neighbor
from src.local_search.two_opt import local_search_2opt
from src.grasp.grasp import run_grasp
from src.exact.branch_and_bound import BranchAndBoundTSP

def generate_random_instance(n, width=100, height=100):
    """Génère n points aléatoires dans un rectangle."""
    coords = []
    for _ in range(n):
        coords.append((random.uniform(0, width), random.uniform(0, height)))
    
    # Matrice Euclidienne
    matrix = build_distance_matrix(coords, "EUC_2D")
    return coords, matrix

def run_experiments():
    # === Paramètres ===
    # On teste sur des petites et moyennes instances
    sizes_exact = [4, 6, 8, 10, 11] # Exact explose vite !
    sizes_heuristic = [10, 20, 50, 100]
    
    nb_runs = 3 # Moyenne sur plusieurs runs pour lisser
    
    print(f"{'N':<5} | {'ALGO':<15} | {'COUT MOYEN':<12} | {'TEMPS MOYEN (s)':<15} | {'GAP (%)':<10}")
    print("-" * 75)

    # 1. Comparaison avec EXACT (Petites instances)
    print("--- COMPARAISON EXACT VS HEURISTIQUES (Petites tailles) ---")
    
    for n in sizes_exact:
        t_exact, c_exact = 0, 0
        t_nn, c_nn = 0, 0
        t_ls, c_ls = 0, 0
        t_grasp, c_grasp = 0, 0
        
        for _ in range(nb_runs):
            _, matrix = generate_random_instance(n)
            
            # EXACT
            solver = BranchAndBoundTSP(n, matrix)
            t0 = time.time()
            _, cost = solver.solve() # Hack: output masqué
            t_exact += (time.time() - t0)
            c_exact += cost
            best_known = cost
            
            # NN
            t0 = time.time()
            p = constructive_nearest_neighbor(n, matrix)
            c = calculate_total_cost(p, matrix)
            t_nn += (time.time() - t0)
            c_nn += c
            
            # LS (2-Opt)
            t0 = time.time()
            p = local_search_2opt(p, matrix) # Améliore NN
            c = calculate_total_cost(p, matrix)
            t_ls += (time.time() - t0)
            c_ls += c
            
            # GRASP
            t0 = time.time()
            _, c = run_grasp(n, matrix, max_iterations=10, alpha=2) # Light GRASP
            t_grasp += (time.time() - t0)
            c_grasp += c
            
        # Moyennes
        res = [
            ("Exact", c_exact/nb_runs, t_exact/nb_runs),
            ("Cons(NN)", c_nn/nb_runs, t_nn/nb_runs),
            ("Loc(2Opt)", c_ls/nb_runs, t_ls/nb_runs),
            ("GRASP", c_grasp/nb_runs, t_grasp/nb_runs)
        ]
        
        baseline = res[0][1] # Cout Exact
        
        for name, cost, duration in res:
            gap = ((cost - baseline) / baseline) * 100 if baseline > 0 else 0
            print(f"{n:<5} | {name:<15} | {cost:<12.1f} | {duration:<15.5f} | {gap:<10.1f}%")
        print("-" * 75)

    # 2. Heuristiques seulement (Plus grosses instances)
    print("\n\n--- COMPARAISON HEURISTIQUES SEULEMENT (Tailles moy/grandes) ---")
    print("(Exact désactivé car trop lent)")
    print("-" * 75)
    
    for n in sizes_heuristic:
        t_nn, c_nn = 0, 0
        t_ls, c_ls = 0, 0
        t_grasp, c_grasp = 0, 0
        
        for _ in range(nb_runs):
            _, matrix = generate_random_instance(n)
            
            # NN
            t0 = time.time()
            p = constructive_nearest_neighbor(n, matrix)
            c = calculate_total_cost(p, matrix)
            t_nn += (time.time() - t0)
            c_nn += c
            
            # LS
            t0 = time.time()
            p = local_search_2opt(p, matrix)
            c = calculate_total_cost(p, matrix)
            t_ls += (time.time() - t0)
            c_ls += c
            
            # GRASP
            t0 = time.time()
            _, c = run_grasp(n, matrix, max_iterations=20, alpha=2)
            t_grasp += (time.time() - t0)
            c_grasp += c
            
        # Moyennes
        res = [
            ("Cons(NN)", c_nn/nb_runs, t_nn/nb_runs),
            ("Loc(2Opt)", c_ls/nb_runs, t_ls/nb_runs),
            ("GRASP", c_grasp/nb_runs, t_grasp/nb_runs)
        ]
        
        # On prend GRASP comme référence (souvent le meilleur)
        baseline = res[2][1] 
        
        for name, cost, duration in res:
            gap = ((cost - baseline) / baseline) * 100 if baseline > 0 else 0
            print(f"{n:<5} | {name:<15} | {cost:<12.1f} | {duration:<15.5f} | {gap:<10.1f}%")
        print("-" * 75)

if __name__ == "__main__":
    # Rediriger stdout pour éviter le spam des prints internes des algos
    # On va juste afficher le tableau final
    original_stdout = sys.stdout
    with open('experiments_log.txt', 'w') as f:
        sys.stdout = f
        try:
            print("Lancement des expérimentations... (Logs complets)")
            run_experiments()
        except Exception as e:
             print(f"Erreur: {e}")
        finally:
            sys.stdout = original_stdout
            
    print("Expérimentations terminées. Voir experiments_log.txt pour les détails.")
