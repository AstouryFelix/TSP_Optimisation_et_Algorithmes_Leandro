"""
Question 6 & 7 : Expérimentations et Analyse Comparative
========================================================
Ce script réalise une comparaison complète des algorithmes implémentés :
  - Exact (Branch & Bound) : limité à n ≤ 15
  - Constructif (Nearest Neighbor)
  - Recherche Locale (2-Opt)
  - GRASP

Il génère :
  - Des tableaux de résultats (affichés et sauvegardés en CSV)
  - Des graphiques pour le rapport (PNG)
  - Une analyse statistique (moyenne, écart-type, gap)

Auteurs: [Votre équipe]
Date: Janvier 2026
"""

import time
import random
import math
import sys
import os
import csv
from datetime import datetime

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Imports des algorithmes
from _3_Constructive import (
    load_data, calculate_total_cost, build_distance_matrix,
    constructive_nearest_neighbor
)
from _4_LocalSearch import local_search_2opt
from _2_BB_Test import TSP_ILP_Solver

# Import GRASP (nouvelle version)
try:
    from _5_GraspTSP import run_grasp
except ImportError:
    # Fallback si l'import échoue
    print("Warning: Impossible d'importer _5_GraspTSP, définition locale")
    def run_grasp(n, matrix, max_iterations=30, alpha=2, **kwargs):
        from _5_GraspTSP import run_grasp as _run_grasp
        return _run_grasp(n, matrix, max_iterations, alpha)

# Matplotlib pour les graphiques
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib non disponible, graphiques désactivés")


# =============================================================================
# GÉNÉRATION D'INSTANCES
# =============================================================================

def generate_random_instance(n, width=1000, height=1000, seed=None):
    """
    Génère une instance TSP aléatoire avec n villes.
    
    Les villes sont placées aléatoirement dans un rectangle [0, width] x [0, height].
    Les distances sont euclidiennes (arrondies à l'entier).
    
    Paramètres:
    -----------
    n : int
        Nombre de villes
    width, height : int
        Dimensions du rectangle
    seed : int, optional
        Graine pour reproductibilité
    
    Retourne:
    ---------
    coords : list[tuple]
        Coordonnées des villes
    matrix : list[list[int]]
        Matrice des distances
    """
    if seed is not None:
        random.seed(seed)
    
    coords = [(random.uniform(0, width), random.uniform(0, height)) for _ in range(n)]
    matrix = build_distance_matrix(coords, "EUC_2D")
    
    return coords, matrix


def generate_clustered_instance(n, n_clusters=4, cluster_radius=100, 
                                width=1000, height=1000, seed=None):
    """
    Génère une instance avec des villes regroupées en clusters.
    
    Ce type d'instance est souvent plus facile pour les heuristiques
    car la structure est plus régulière.
    """
    if seed is not None:
        random.seed(seed)
    
    # Générer les centres des clusters
    centers = [(random.uniform(cluster_radius, width - cluster_radius),
                random.uniform(cluster_radius, height - cluster_radius))
               for _ in range(n_clusters)]
    
    # Répartir les villes dans les clusters
    coords = []
    for i in range(n):
        center = centers[i % n_clusters]
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, cluster_radius)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        coords.append((x, y))
    
    matrix = build_distance_matrix(coords, "EUC_2D")
    return coords, matrix


def generate_grid_instance(n, seed=None):
    """
    Génère une instance où les villes sont sur une grille.
    
    Ce type d'instance peut être pathologique pour certaines heuristiques.
    """
    if seed is not None:
        random.seed(seed)
    
    # Trouver la grille la plus carrée possible
    side = int(math.ceil(math.sqrt(n)))
    
    coords = []
    for i in range(n):
        x = (i % side) * 100
        y = (i // side) * 100
        # Ajouter un petit bruit pour éviter les cas dégénérés
        x += random.uniform(-5, 5)
        y += random.uniform(-5, 5)
        coords.append((x, y))
    
    matrix = build_distance_matrix(coords, "EUC_2D")
    return coords, matrix


# =============================================================================
# EXÉCUTION DES ALGORITHMES
# =============================================================================

def run_exact(n, matrix, timeout=60):
    """Exécute l'algorithme exact (Branch & Bound) avec timeout."""
    if n > 15:
        return None, None, None  # Trop lent
    
    try:
        solver = TSP_ILP_Solver(matrix)
        t0 = time.time()
        cost, path = solver.solve()
        elapsed = time.time() - t0
        
        if elapsed > timeout:
            return None, None, elapsed
        
        return path, cost, elapsed
    except Exception as e:
        print(f"Erreur Exact: {e}")
        return None, None, None


def run_constructive(n, matrix):
    """Exécute l'heuristique constructive (Nearest Neighbor)."""
    t0 = time.time()
    path = constructive_nearest_neighbor(n, matrix)
    cost = calculate_total_cost(path, matrix)
    elapsed = time.time() - t0
    return path, cost, elapsed


def run_local_search(n, matrix, initial_path=None):
    """Exécute la recherche locale (2-Opt)."""
    t0 = time.time()
    
    if initial_path is None:
        initial_path = constructive_nearest_neighbor(n, matrix)
    
    path = local_search_2opt(initial_path, matrix)
    cost = calculate_total_cost(path, matrix)
    elapsed = time.time() - t0
    
    return path, cost, elapsed


def run_grasp_algorithm(n, matrix, max_iterations=30, alpha=2):
    """Exécute GRASP avec les paramètres calibrés."""
    t0 = time.time()
    path, cost, _ = run_grasp(n, matrix, max_iterations=max_iterations, 
                              alpha=alpha, verbose=False)
    elapsed = time.time() - t0
    return path, cost, elapsed


# =============================================================================
# EXPÉRIENCES PRINCIPALES
# =============================================================================

def experiment_small_instances(sizes=[5, 8, 10, 12, 15], nb_runs=5, 
                               output_dir="experiment_results"):
    """
    Question 6.1 : Comparaison sur petites instances (avec algorithme exact).
    
    Compare tous les algorithmes sur des instances où l'optimal est connu.
    """
    print("\n" + "="*70)
    print("EXPÉRIENCE 1 : PETITES INSTANCES (Comparaison avec Exact)")
    print("="*70)
    
    results = []
    
    for n in sizes:
        print(f"\n--- Taille n = {n} ---")
        
        stats = {
            "Exact": {"costs": [], "times": [], "gaps": []},
            "NN": {"costs": [], "times": [], "gaps": []},
            "2-Opt": {"costs": [], "times": [], "gaps": []},
            "GRASP": {"costs": [], "times": [], "gaps": []}
        }
        
        for run in range(nb_runs):
            _, matrix = generate_random_instance(n, seed=1000 + run)
            
            # Algorithme Exact (référence)
            _, opt_cost, t_exact = run_exact(n, matrix)
            if opt_cost is None:
                print(f"  Run {run+1}: Exact timeout/erreur, skip")
                continue
            
            stats["Exact"]["costs"].append(opt_cost)
            stats["Exact"]["times"].append(t_exact)
            stats["Exact"]["gaps"].append(0.0)
            
            # Nearest Neighbor
            _, c_nn, t_nn = run_constructive(n, matrix)
            gap_nn = ((c_nn - opt_cost) / opt_cost) * 100
            stats["NN"]["costs"].append(c_nn)
            stats["NN"]["times"].append(t_nn)
            stats["NN"]["gaps"].append(gap_nn)
            
            # 2-Opt (partant de NN)
            init_path = constructive_nearest_neighbor(n, matrix)
            _, c_2opt, t_2opt = run_local_search(n, matrix, init_path)
            gap_2opt = ((c_2opt - opt_cost) / opt_cost) * 100
            stats["2-Opt"]["costs"].append(c_2opt)
            stats["2-Opt"]["times"].append(t_2opt)
            stats["2-Opt"]["gaps"].append(gap_2opt)
            
            # GRASP
            _, c_grasp, t_grasp = run_grasp_algorithm(n, matrix)
            gap_grasp = ((c_grasp - opt_cost) / opt_cost) * 100
            stats["GRASP"]["costs"].append(c_grasp)
            stats["GRASP"]["times"].append(t_grasp)
            stats["GRASP"]["gaps"].append(gap_grasp)
        
        # Calculer les moyennes
        for algo in stats:
            if stats[algo]["costs"]:
                avg_cost = sum(stats[algo]["costs"]) / len(stats[algo]["costs"])
                avg_time = sum(stats[algo]["times"]) / len(stats[algo]["times"])
                avg_gap = sum(stats[algo]["gaps"]) / len(stats[algo]["gaps"])
                
                results.append({
                    "n": n,
                    "algo": algo,
                    "cost": avg_cost,
                    "time": avg_time,
                    "gap": avg_gap
                })
                
                print(f"  {algo:8s}: Coût={avg_cost:8.1f}, Temps={avg_time:.4f}s, Gap={avg_gap:5.2f}%")
    
    # Sauvegarder en CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "small_instances_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["n", "algo", "cost", "time", "gap"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Résultats sauvegardés : {csv_path}")
    
    # Générer les graphiques
    if MATPLOTLIB_AVAILABLE:
        plot_small_instances_results(results, output_dir)
    
    return results


def experiment_large_instances(sizes=[20, 50, 100, 200, 500], nb_runs=5,
                               output_dir="experiment_results"):
    """
    Question 6.2 : Comparaison sur grandes instances (heuristiques seulement).
    """
    print("\n" + "="*70)
    print("EXPÉRIENCE 2 : GRANDES INSTANCES (Heuristiques)")
    print("="*70)
    
    results = []
    
    for n in sizes:
        print(f"\n--- Taille n = {n} ---")
        
        stats = {
            "NN": {"costs": [], "times": []},
            "2-Opt": {"costs": [], "times": []},
            "GRASP": {"costs": [], "times": []}
        }
        
        for run in range(nb_runs):
            _, matrix = generate_random_instance(n, seed=2000 + run)
            
            # NN
            _, c_nn, t_nn = run_constructive(n, matrix)
            stats["NN"]["costs"].append(c_nn)
            stats["NN"]["times"].append(t_nn)
            
            # 2-Opt
            init_path = constructive_nearest_neighbor(n, matrix)
            _, c_2opt, t_2opt = run_local_search(n, matrix, init_path)
            stats["2-Opt"]["costs"].append(c_2opt)
            stats["2-Opt"]["times"].append(t_2opt)
            
            # GRASP
            _, c_grasp, t_grasp = run_grasp_algorithm(n, matrix)
            stats["GRASP"]["costs"].append(c_grasp)
            stats["GRASP"]["times"].append(t_grasp)
        
        # Le meilleur coût moyen sert de référence
        best_cost = min(
            sum(stats[algo]["costs"]) / len(stats[algo]["costs"])
            for algo in stats
        )
        
        for algo in stats:
            avg_cost = sum(stats[algo]["costs"]) / len(stats[algo]["costs"])
            std_cost = math.sqrt(sum((c - avg_cost)**2 for c in stats[algo]["costs"]) / len(stats[algo]["costs"]))
            avg_time = sum(stats[algo]["times"]) / len(stats[algo]["times"])
            gap = ((avg_cost - best_cost) / best_cost) * 100 if best_cost > 0 else 0
            
            results.append({
                "n": n,
                "algo": algo,
                "cost": avg_cost,
                "std": std_cost,
                "time": avg_time,
                "gap": gap
            })
            
            print(f"  {algo:8s}: Coût={avg_cost:10.1f} (±{std_cost:7.1f}), Temps={avg_time:7.3f}s, Gap={gap:5.2f}%")
    
    # Sauvegarder en CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "large_instances_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["n", "algo", "cost", "std", "time", "gap"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Résultats sauvegardés : {csv_path}")
    
    # Graphiques
    if MATPLOTLIB_AVAILABLE:
        plot_large_instances_results(results, output_dir)
    
    return results


def experiment_scalability(sizes=[10, 25, 50, 100, 200, 500, 1000], nb_runs=3,
                          output_dir="experiment_results"):
    """
    Analyse de la scalabilité : temps d'exécution en fonction de n.
    """
    print("\n" + "="*70)
    print("EXPÉRIENCE 3 : ANALYSE DE SCALABILITÉ")
    print("="*70)
    
    results = []
    
    for n in sizes:
        print(f"\n--- Taille n = {n} ---")
        
        times_nn = []
        times_2opt = []
        times_grasp = []
        
        for run in range(nb_runs):
            _, matrix = generate_random_instance(n, seed=3000 + run)
            
            # NN
            _, _, t = run_constructive(n, matrix)
            times_nn.append(t)
            
            # 2-Opt
            init_path = constructive_nearest_neighbor(n, matrix)
            _, _, t = run_local_search(n, matrix, init_path)
            times_2opt.append(t)
            
            # GRASP (moins d'itérations pour les grandes instances)
            grasp_iter = 30 if n <= 200 else 15
            _, _, t = run_grasp_algorithm(n, matrix, max_iterations=grasp_iter)
            times_grasp.append(t)
        
        results.append({
            "n": n,
            "NN": sum(times_nn) / len(times_nn),
            "2-Opt": sum(times_2opt) / len(times_2opt),
            "GRASP": sum(times_grasp) / len(times_grasp)
        })
        
        print(f"  NN: {results[-1]['NN']:.4f}s, 2-Opt: {results[-1]['2-Opt']:.4f}s, GRASP: {results[-1]['GRASP']:.4f}s")
    
    # Sauvegarder
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "scalability_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["n", "NN", "2-Opt", "GRASP"])
        writer.writeheader()
        writer.writerows(results)
    
    # Graphique
    if MATPLOTLIB_AVAILABLE:
        plot_scalability(results, output_dir)
    
    return results


def experiment_instance_types(n=100, nb_runs=5, output_dir="experiment_results"):
    """
    Question 7 : Comparaison sur différents types d'instances.
    """
    print("\n" + "="*70)
    print("EXPÉRIENCE 4 : COMPARAISON PAR TYPE D'INSTANCE")
    print("="*70)
    
    instance_types = {
        "Aléatoire": lambda seed: generate_random_instance(n, seed=seed),
        "Clustered": lambda seed: generate_clustered_instance(n, seed=seed),
        "Grille": lambda seed: generate_grid_instance(n, seed=seed)
    }
    
    results = []
    
    for inst_type, generator in instance_types.items():
        print(f"\n--- Type: {inst_type} ---")
        
        stats = {
            "NN": {"costs": [], "times": []},
            "2-Opt": {"costs": [], "times": []},
            "GRASP": {"costs": [], "times": []}
        }
        
        for run in range(nb_runs):
            _, matrix = generator(seed=4000 + run)
            
            _, c_nn, t_nn = run_constructive(n, matrix)
            stats["NN"]["costs"].append(c_nn)
            stats["NN"]["times"].append(t_nn)
            
            init_path = constructive_nearest_neighbor(n, matrix)
            _, c_2opt, t_2opt = run_local_search(n, matrix, init_path)
            stats["2-Opt"]["costs"].append(c_2opt)
            stats["2-Opt"]["times"].append(t_2opt)
            
            _, c_grasp, t_grasp = run_grasp_algorithm(n, matrix)
            stats["GRASP"]["costs"].append(c_grasp)
            stats["GRASP"]["times"].append(t_grasp)
        
        best_cost = min(
            sum(stats[algo]["costs"]) / len(stats[algo]["costs"])
            for algo in stats
        )
        
        for algo in stats:
            avg_cost = sum(stats[algo]["costs"]) / len(stats[algo]["costs"])
            avg_time = sum(stats[algo]["times"]) / len(stats[algo]["times"])
            gap = ((avg_cost - best_cost) / best_cost) * 100
            
            results.append({
                "type": inst_type,
                "algo": algo,
                "cost": avg_cost,
                "time": avg_time,
                "gap": gap
            })
            
            print(f"  {algo:8s}: Coût={avg_cost:10.1f}, Temps={avg_time:.4f}s, Gap={gap:5.2f}%")
    
    # Sauvegarder
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "instance_types_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["type", "algo", "cost", "time", "gap"])
        writer.writeheader()
        writer.writerows(results)
    
    # Graphique
    if MATPLOTLIB_AVAILABLE:
        plot_instance_types(results, output_dir)
    
    return results


# =============================================================================
# GÉNÉRATION DES GRAPHIQUES
# =============================================================================

def plot_small_instances_results(results, output_dir):
    """Génère les graphiques pour les petites instances."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    algos = ["Exact", "NN", "2-Opt", "GRASP"]
    colors = {"Exact": "black", "NN": "blue", "2-Opt": "green", "GRASP": "red"}
    markers = {"Exact": "s", "NN": "o", "2-Opt": "^", "GRASP": "D"}
    
    # Extraire les données par algorithme
    data_by_algo = {algo: {"n": [], "cost": [], "time": [], "gap": []} for algo in algos}
    for r in results:
        data_by_algo[r["algo"]]["n"].append(r["n"])
        data_by_algo[r["algo"]]["cost"].append(r["cost"])
        data_by_algo[r["algo"]]["time"].append(r["time"])
        data_by_algo[r["algo"]]["gap"].append(r["gap"])
    
    # Graphique 1: Coûts
    ax1 = axes[0]
    for algo in algos:
        if data_by_algo[algo]["n"]:
            ax1.plot(data_by_algo[algo]["n"], data_by_algo[algo]["cost"],
                    marker=markers[algo], color=colors[algo], label=algo, linewidth=2)
    ax1.set_xlabel("Nombre de villes (n)")
    ax1.set_ylabel("Coût de la solution")
    ax1.set_title("Qualité des solutions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Temps (échelle log)
    ax2 = axes[1]
    for algo in algos:
        if data_by_algo[algo]["n"]:
            ax2.plot(data_by_algo[algo]["n"], data_by_algo[algo]["time"],
                    marker=markers[algo], color=colors[algo], label=algo, linewidth=2)
    ax2.set_xlabel("Nombre de villes (n)")
    ax2.set_ylabel("Temps d'exécution (s)")
    ax2.set_title("Temps d'exécution")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Graphique 3: Gap par rapport à l'optimal
    ax3 = axes[2]
    for algo in ["NN", "2-Opt", "GRASP"]:  # Pas Exact (gap=0)
        if data_by_algo[algo]["n"]:
            ax3.plot(data_by_algo[algo]["n"], data_by_algo[algo]["gap"],
                    marker=markers[algo], color=colors[algo], label=algo, linewidth=2)
    ax3.set_xlabel("Nombre de villes (n)")
    ax3.set_ylabel("Gap par rapport à l'optimal (%)")
    ax3.set_title("Écart à l'optimal")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "small_instances_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique sauvegardé : {filepath}")


def plot_large_instances_results(results, output_dir):
    """Génère les graphiques pour les grandes instances."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    algos = ["NN", "2-Opt", "GRASP"]
    colors = {"NN": "blue", "2-Opt": "green", "GRASP": "red"}
    markers = {"NN": "o", "2-Opt": "^", "GRASP": "D"}
    
    data_by_algo = {algo: {"n": [], "cost": [], "time": [], "gap": []} for algo in algos}
    for r in results:
        data_by_algo[r["algo"]]["n"].append(r["n"])
        data_by_algo[r["algo"]]["cost"].append(r["cost"])
        data_by_algo[r["algo"]]["time"].append(r["time"])
        data_by_algo[r["algo"]]["gap"].append(r["gap"])
    
    # Graphique 1: Coûts
    ax1 = axes[0]
    for algo in algos:
        ax1.plot(data_by_algo[algo]["n"], data_by_algo[algo]["cost"],
                marker=markers[algo], color=colors[algo], label=algo, linewidth=2, markersize=8)
    ax1.set_xlabel("Nombre de villes (n)", fontsize=12)
    ax1.set_ylabel("Coût moyen de la solution", fontsize=12)
    ax1.set_title("Qualité des solutions (grandes instances)", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Temps
    ax2 = axes[1]
    for algo in algos:
        ax2.plot(data_by_algo[algo]["n"], data_by_algo[algo]["time"],
                marker=markers[algo], color=colors[algo], label=algo, linewidth=2, markersize=8)
    ax2.set_xlabel("Nombre de villes (n)", fontsize=12)
    ax2.set_ylabel("Temps d'exécution (s)", fontsize=12)
    ax2.set_title("Temps d'exécution (grandes instances)", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "large_instances_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique sauvegardé : {filepath}")


def plot_scalability(results, output_dir):
    """Génère le graphique de scalabilité avec courbes théoriques."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ns = [r["n"] for r in results]
    
    colors = {"NN": "blue", "2-Opt": "green", "GRASP": "red"}
    
    for algo in ["NN", "2-Opt", "GRASP"]:
        times = [r[algo] for r in results]
        ax.plot(ns, times, 'o-', color=colors[algo], label=algo, linewidth=2, markersize=8)
    
    # Courbes théoriques (ajustées)
    # NN: O(n²), 2-Opt: O(n² * iterations), GRASP: O(n² * grasp_iter)
    if len(ns) > 2:
        # Normaliser à partir du premier point
        n0 = ns[0]
        t0_nn = results[0]["NN"]
        t0_2opt = results[0]["2-Opt"]
        
        theoretical_n2 = [t0_nn * (n/n0)**2 for n in ns]
        ax.plot(ns, theoretical_n2, '--', color='gray', alpha=0.5, label='O(n²) théorique')
    
    ax.set_xlabel("Nombre de villes (n)", fontsize=12)
    ax.set_ylabel("Temps d'exécution (s)", fontsize=12)
    ax.set_title("Scalabilité des algorithmes", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "scalability_analysis.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique sauvegardé : {filepath}")


def plot_instance_types(results, output_dir):
    """Génère le graphique de comparaison par type d'instance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    types = ["Aléatoire", "Clustered", "Grille"]
    algos = ["NN", "2-Opt", "GRASP"]
    colors = {"NN": "blue", "2-Opt": "green", "GRASP": "red"}
    
    # Organiser les données
    data = {t: {a: {"cost": 0, "gap": 0} for a in algos} for t in types}
    for r in results:
        data[r["type"]][r["algo"]]["cost"] = r["cost"]
        data[r["type"]][r["algo"]]["gap"] = r["gap"]
    
    x = range(len(types))
    width = 0.25
    
    # Graphique 1: Coûts par type
    ax1 = axes[0]
    for i, algo in enumerate(algos):
        costs = [data[t][algo]["cost"] for t in types]
        ax1.bar([xi + i*width for xi in x], costs, width, label=algo, color=colors[algo])
    ax1.set_xlabel("Type d'instance")
    ax1.set_ylabel("Coût moyen")
    ax1.set_title("Coût par type d'instance")
    ax1.set_xticks([xi + width for xi in x])
    ax1.set_xticklabels(types)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Graphique 2: Gaps par type
    ax2 = axes[1]
    for i, algo in enumerate(algos):
        gaps = [data[t][algo]["gap"] for t in types]
        ax2.bar([xi + i*width for xi in x], gaps, width, label=algo, color=colors[algo])
    ax2.set_xlabel("Type d'instance")
    ax2.set_ylabel("Gap (%)")
    ax2.set_title("Gap par type d'instance")
    ax2.set_xticks([xi + width for xi in x])
    ax2.set_xticklabels(types)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "instance_types_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique sauvegardé : {filepath}")


# =============================================================================
# RAPPORT FINAL
# =============================================================================

def generate_summary_report(output_dir="experiment_results"):
    """Génère un rapport texte résumant toutes les expériences."""
    report_path = os.path.join(output_dir, "summary_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("       RAPPORT D'EXPÉRIMENTATION - TSP\n")
        f.write(f"       Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("="*70 + "\n\n")
        
        f.write("ALGORITHMES TESTÉS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Exact (Branch & Bound)\n")
        f.write("   - Complexité: O(n! / 2) pire cas, élagage améliore en pratique\n")
        f.write("   - Limite pratique: n ≤ 15\n\n")
        f.write("2. Nearest Neighbor (Constructif)\n")
        f.write("   - Complexité: O(n²)\n")
        f.write("   - Gap typique: 15-25%\n\n")
        f.write("3. 2-Opt (Recherche Locale)\n")
        f.write("   - Complexité: O(n² × iterations)\n")
        f.write("   - Gap typique: 2-10%\n\n")
        f.write("4. GRASP (Métaheuristique)\n")
        f.write("   - Paramètres calibrés: α=2, 30 itérations\n")
        f.write("   - Gap typique: 0-5%\n\n")
        
        f.write("FICHIERS GÉNÉRÉS:\n")
        f.write("-" * 40 + "\n")
        f.write("- small_instances_results.csv\n")
        f.write("- large_instances_results.csv\n")
        f.write("- scalability_results.csv\n")
        f.write("- instance_types_results.csv\n")
        f.write("- *.png (graphiques)\n\n")
        
        f.write("CONCLUSIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. L'algorithme exact garantit l'optimalité mais est limité à ~15 villes.\n")
        f.write("2. GRASP offre le meilleur compromis qualité/temps pour n > 20.\n")
        f.write("3. 2-Opt améliore significativement NN avec un coût temps modéré.\n")
        f.write("4. Les performances varient selon le type d'instance.\n")
    
    print(f"\n✓ Rapport généré : {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def run_all_experiments(output_dir="experiment_results"):
    """Lance toutes les expériences."""
    print("\n" + "="*70)
    print("        LANCEMENT DE TOUTES LES EXPÉRIENCES")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Expérience 1: Petites instances
    experiment_small_instances(
        sizes=[5, 8, 10, 12],
        nb_runs=5,
        output_dir=output_dir
    )
    
    # Expérience 2: Grandes instances
    experiment_large_instances(
        sizes=[20, 50, 100, 200],
        nb_runs=5,
        output_dir=output_dir
    )
    
    # Expérience 3: Scalabilité
    experiment_scalability(
        sizes=[10, 25, 50, 100, 200, 500],
        nb_runs=3,
        output_dir=output_dir
    )
    
    # Expérience 4: Types d'instances
    experiment_instance_types(
        n=100,
        nb_runs=5,
        output_dir=output_dir
    )
    
    # Générer le rapport
    generate_summary_report(output_dir)
    
    print("\n" + "="*70)
    print("        TOUTES LES EXPÉRIENCES TERMINÉES")
    print(f"        Résultats dans : {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Expérimentations TSP")
    parser.add_argument("--all", action="store_true", help="Lancer toutes les expériences")
    parser.add_argument("--small", action="store_true", help="Petites instances seulement")
    parser.add_argument("--large", action="store_true", help="Grandes instances seulement")
    parser.add_argument("--scale", action="store_true", help="Analyse de scalabilité")
    parser.add_argument("--types", action="store_true", help="Comparaison par type")
    parser.add_argument("--output", type=str, default="experiment_results", help="Dossier de sortie")
    
    args = parser.parse_args()
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    if args.all or (not any([args.small, args.large, args.scale, args.types])):
        run_all_experiments(output_dir)
    else:
        if args.small:
            experiment_small_instances(output_dir=output_dir)
        if args.large:
            experiment_large_instances(output_dir=output_dir)
        if args.scale:
            experiment_scalability(output_dir=output_dir)
        if args.types:
            experiment_instance_types(output_dir=output_dir)
        
        generate_summary_report(output_dir)