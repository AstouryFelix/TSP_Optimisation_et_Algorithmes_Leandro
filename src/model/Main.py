"""
Main Script for TSP Comparison
==============================
Ce script permet d'exécuter tous les algorithmes sur UN fichier d'instance donné (.in ou .tsp)
et de comparer leurs performances (Coût et Temps).

Algorithmes :
1. Exact (Branch & Bound) - Limité par défaut à N <= 20
2. Constructif (Nearest Neighbor)
3. Recherche Locale (2-Opt)
4. GRASP

Usage:
    python Programmes/Main.py data/Input/instance.in
    python Programmes/Main.py data/Input/instance.tsp --force-exact
"""

import sys
import os
import time
import argparse

# Ajout du dossier courant au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Tools.load_data import load_data
from _2_BB_Test import TSP_ILP_Solver
from _3_Constructive import constructive_nearest_neighbor, calculate_total_cost
from _4_LocalSearch import local_search_2opt
from _5_GraspTSP import run_grasp

def run_comparison(file_path, force_exact=False):
    if not os.path.exists(file_path):
        print(f"Erreur : Fichier introuvable '{file_path}'")
        return

    print(f"\n{'='*60}")
    print(f"ANALYSE DE L'INSTANCE : {os.path.basename(file_path)}")
    print(f"{'='*60}")

    # 1. Chargement
    try:
        n, matrix = load_data(file_path)
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        return

    print(f"Nombre de villes (N) : {n}")
    print(f"Format de fichier    : {file_path.split('.')[-1]}")
    print("-" * 60)
    print(f"{'ALGORITHME':<20} | {'COÛT':<15} | {'TEMPS (s)':<10} | {'GAP (%)':<10}")
    print("-" * 60)

    results = {}
    best_cost_found = float('inf')

    
    # Stockage des résultats pour affichage final
    # Structure: {'AlgoName': {'cost': float, 'time': float, 'path': list}}
    
    # --- 1. Algorithme Exact (Branch & Bound) ---
    if n <= 12 or force_exact:
        print(f"  > Exécution Exact (Max 60s)...")
        try:
            t0 = time.time()
            solver = TSP_ILP_Solver(matrix)
            cost, path = solver.solve(verbose=False)
            t1 = time.time()
            
            bb_time = t1 - t0
            results['Exact (B&B)'] = {'cost': cost, 'time': bb_time}
            best_cost_found = min(best_cost_found, cost)
            
            # Timeout warning
            if bb_time > 60:
                print(f"    ! Attention: Temps d'exécution long ({bb_time:.2f}s)")
                
        except Exception as e:
            results['Exact (B&B)'] = {'error': str(e)}
    else:
        results['Exact (B&B)'] = {'skip': True} # N > 12

    # --- 2. Constructif (Nearest Neighbor) ---
    print(f"  > Exécution Nearest Neighbor...")
    try:
        t0 = time.time()
        path_nn = constructive_nearest_neighbor(n, matrix)
        cost_nn = calculate_total_cost(path_nn, matrix)
        t1 = time.time()
        
        nn_time = t1 - t0
        results['Nearest Neighbor'] = {'cost': cost_nn, 'time': nn_time, 'path': path_nn}
        best_cost_found = min(best_cost_found, cost_nn)
        
    except Exception as e:
        results['Nearest Neighbor'] = {'error': str(e)}

    # --- 3. Recherche Locale (2-Opt) ---
    print(f"  > Exécution 2-Opt...")
    try:
        # On repart de la solution NN
        initial_path = results.get('Nearest Neighbor', {}).get('path')
        if not initial_path:
            initial_path = constructive_nearest_neighbor(n, matrix)
            
        t0 = time.time()
        path_2opt = local_search_2opt(initial_path, matrix)
        cost_2opt = calculate_total_cost(path_2opt, matrix)
        t1 = time.time()
        
        time_2opt = t1 - t0
        results['2-Opt'] = {'cost': cost_2opt, 'time': time_2opt}
        best_cost_found = min(best_cost_found, cost_2opt)
        
    except Exception as e:
        results['2-Opt'] = {'error': str(e)}

    # --- 4. GRASP ---
    print(f"  > Exécution GRASP...")
    try:
        t0 = time.time()
        # Paramètres par défaut : alpha=2, max_iter=30
        path_grasp, cost_grasp, _ = run_grasp(n, matrix, max_iterations=30, alpha=2, verbose=False)
        t1 = time.time()
        
        time_grasp = t1 - t0
        results['GRASP'] = {'cost': cost_grasp, 'time': time_grasp}
        best_cost_found = min(best_cost_found, cost_grasp)
        
    except Exception as e:
        results['GRASP'] = {'error': str(e)}
        
    # --- AFFICHAGE DU TABLEAU FINAL ---
    print("\n" + "-" * 80)
    print(f"{'ALGORITHME':<20} | {'COÛT':<15} | {'TEMPS (s)':<10} | {'GAP (%)':<10}")
    print("-" * 80)
    
    # Ordre d'affichage
    algos_order = ['Exact (B&B)', 'Nearest Neighbor', '2-Opt', 'GRASP']
    
    # Si on a un résultat Exact, c'est la référence absolue
    ref_cost = best_cost_found
    if 'Exact (B&B)' in results and 'cost' in results['Exact (B&B)']:
         ref_cost = results['Exact (B&B)']['cost']
    
    for algo in algos_order:
        res = results.get(algo, {})
        
        if 'error' in res:
            print(f"{algo:<20} | {'ERREUR':<15} | {res['error']}")
        elif 'skip' in res:
            print(f"{algo:<20} | {'SKIP (N>12)':<15} | {'-':<10} | {'-':<10}")
        elif 'cost' in res:
            c = res['cost']
            t = res['time']
            # Gap calculé par rapport à la meilleure solution trouvée (ou Exact)
            gap = ((c - ref_cost) / ref_cost * 100) if ref_cost > 0 else 0
            
            print(f"{algo:<20} | {c:<15.1f} | {t:<10.4f} | {gap:<10.2f}")
            
    print("-" * 80)
    print(f"Meilleur coût trouvé : {best_cost_found}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparer les algorithmes TSP sur un fichier.")
    parser.add_argument("file", help="Chemin vers le fichier d'instance (.in ou .tsp)")
    parser.add_argument("--force-exact", action="store_true", help="Forcer l'exécution de l'algorthme Exact même pour N > 20")
    
    args = parser.parse_args()
    
    run_comparison(args.file, args.force_exact)
