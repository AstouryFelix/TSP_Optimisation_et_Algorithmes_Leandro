"""
Script de Benchmark et Génération
=================================
Génère des fichiers d'instances aléatoires (.in) pour N allant de N_START à N_END.
Exécute ensuite Main.run_comparison sur chaque fichier.

Usage:
    python Programmes/_7_Benchmark.py --start 5 --end 20 --step 1
"""

import os
import random
import sys
import argparse
import time

# Import de Main pour utiliser run_comparison
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Main import run_comparison

def generate_random_in_file(n, filename, width=1000, height=1000):
    """Génère un fichier .in avec n villes aléatoires."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        # Format .in: chaque ligne = une ligne de la matrice ? 
        # Non, load_data.py read_instance_in lit une MATRICE.
        # Donc on doit générer une matrice de distances.
        
        # Génération coordonnées
        coords = [(random.uniform(0, width), random.uniform(0, height)) for _ in range(n)]
        
        # Calcul matrice distances
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 0
                else:
                    d = int(((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2)**0.5)
                    matrix[i][j] = d
        
        # Ecriture matrice
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")
            
    return filename

def run_benchmark(start, end, step):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Benchmark")
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print(f"LANCEMENT DU BENCHMARK (N={start} -> {end})")
    print("="*70)
    
    import csv
    
    csv_file = os.path.join(output_dir, "benchmark_results.csv")
    csv_columns = ["n", "algo", "cost", "time", "gap"]
    
    csv_data = []
    summary = []

    for n in range(start, end + 1, step):
        filename = os.path.join(output_dir, f"random_{n}.in")
        generate_random_in_file(n, filename)
        
        print(f"\n--- Instance générée : {filename} (N={n}) ---")
        
        # Lancer la comparaison (Main.py) - on force l'exact jusqu'à 12
        force_exact = (n <= 12)
        
        # Capture des résultats retournés par Main
        results = run_comparison(filename, force_exact=force_exact)
        
        # Déterminer le meilleur coût global pour le calcul du GAP correct
        best_overall_cost = float('inf')
        for algo, data in results.items():
            if 'cost' in data and data['cost'] < best_overall_cost:
                best_overall_cost = data['cost']

        best_algo_for_summary = "N/A"
        min_cost_for_summary = float('inf')
        
        for algo, data in results.items():
            cost = data.get('cost', float('inf'))
            time_val = data.get('time', 0)
            
            # Pour le résumé console
            if cost < min_cost_for_summary:
                min_cost_for_summary = cost
                best_algo_for_summary = algo
            
            # Calcul du gap
            gap = 0.0
            if 'cost' in data and best_overall_cost > 0:
                gap = ((data['cost'] - best_overall_cost) / best_overall_cost) * 100
                
            if 'cost' in data: # On n'enregistre que si l'algo a tourné
                csv_data.append({
                    "n": n,
                    "algo": algo,
                    "cost": cost,
                    "time": time_val,
                    "gap": gap
                })
                
        summary.append({
            'n': n,
            'best_algo': best_algo_for_summary,
            'cost': min_cost_for_summary
        })
        
    # Ecriture du CSV
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_data:
                writer.writerow(data)
        print(f"\n[INFO] Résultats détaillés exportés dans : {csv_file}")
    except Exception as e:
        print(f"[ERREUR] Impossible d'écrire le CSV : {e}")

    print("\n" + "="*70)
    print("RÉSUMÉ DU BENCHMARK")
    print("="*70)
    print(f"{'N':<5} | {'Meilleur Algo':<20} | {'Coût':<10}")
    print("-" * 40)
    for item in summary:
        print(f"{item['n']:<5} | {item['best_algo']:<20} | {item['cost']:<10.1f}")
        
    # Génération des graphiques
    plot_benchmark_results(csv_file)

def plot_benchmark_results(csv_file):
    """Génère des graphiques à partir du fichier CSV."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[WARNING] Matplotlib ou Pandas manquant. Impossible de générer les graphiques.")
        return

    print("\n[INFO] Génération des graphiques...")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"[ERREUR] Lecture CSV impossible : {e}")
        return
        
    output_dir = os.path.dirname(csv_file)
    
    # Configuration du style
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    
    algos = df['algo'].unique()
    
    # 1. Graphique des COÛTS
    plt.figure(figsize=(10, 6))
    for algo in algos:
        data = df[df['algo'] == algo]
        plt.plot(data['n'], data['cost'], marker='o', label=algo)
    plt.title("Comparaison des Coûts")
    plt.xlabel("Nombre de villes (N)")
    plt.ylabel("Coût de la solution")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "benchmark_costs.png"))
    plt.close()
    
    # 2. Graphique des TEMPS (Tous)
    plt.figure(figsize=(10, 6))
    for algo in algos:
        data = df[df['algo'] == algo]
        plt.plot(data['n'], data['time'], marker='s', label=algo)
    plt.title("Comparaison des Temps d'exécution (Tous)")
    plt.xlabel("Nombre de villes (N)")
    plt.ylabel("Temps (s)")
    plt.yscale('log') # Log scale souvent utile incluant le BB
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "benchmark_times_all.png"))
    plt.close()
    
    # 3. Graphique des TEMPS (Sans Exact)
    plt.figure(figsize=(10, 6))
    for algo in algos:
        if "Exact" in algo:
            continue
        data = df[df['algo'] == algo]
        plt.plot(data['n'], data['time'], marker='^', label=algo)
    plt.title("Comparaison des Temps (Heuristiques uniquement)")
    plt.xlabel("Nombre de villes (N)")
    plt.ylabel("Temps (s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "benchmark_times_heuristics.png"))
    plt.close()

    # 4. Graphique du GAP
    plt.figure(figsize=(10, 6))
    for algo in algos:
        data = df[df['algo'] == algo]
        plt.plot(data['n'], data['gap'], marker='x', label=algo)
    plt.title("Evolution du GAP (%)")
    plt.xlabel("Nombre de villes (N)")
    plt.ylabel("Gap (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "benchmark_gaps.png"))
    plt.close()
    
    print(f"[INFO] 4 Graphiques générés dans : {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=5, help="Nombre de villes de départ")
    parser.add_argument("--end", type=int, default=20, help="Nombre de villes de fin")
    parser.add_argument("--step", type=int, default=1, help="Pas d'incrément")
    args = parser.parse_args()
    
    run_benchmark(args.start, args.end, args.step)
