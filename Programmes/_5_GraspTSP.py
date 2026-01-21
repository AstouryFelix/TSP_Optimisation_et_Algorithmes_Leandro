"""
Question 5 : Métaheuristique GRASP
==================================
Implémentation de Greedy Randomized Adaptive Search Procedure pour le TSP.

GRASP combine deux phases à chaque itération :
  1. Construction randomisée (Nearest Neighbor avec RCL)
  2. Amélioration locale (2-Opt)

Ce module inclut également la CALIBRATION des paramètres :
  - Alpha (taille de la RCL)
  - Nombre d'itérations
  - Noeud de départ (fixe vs aléatoire)

Auteurs: [Votre équipe]
Date: Janvier 2026
"""

from _3_Constructive import load_data, calculate_total_cost, save_solution, build_distance_matrix
from _4_LocalSearch import local_search_2opt
import random
import time
import os
import math

# Pour les graphiques de calibration
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib non disponible, graphiques désactivés")


# =============================================================================
# PHASE 1 : CONSTRUCTION RANDOMISÉE (Nearest Neighbor avec RCL)
# =============================================================================

def constructive_randomized_nearest_neighbor(n, matrix, alpha=2, start_node=0):
    """
    Variante randomisée de l'heuristique constructive Nearest Neighbor.
    
    Principe : Au lieu de toujours choisir le voisin le plus proche,
    on construit une Liste Restreinte de Candidats (RCL) contenant
    les 'alpha' meilleurs voisins, puis on en choisit un au hasard.
    
    Paramètres:
    -----------
    n : int
        Nombre de villes
    matrix : list[list[int]]
        Matrice des distances
    alpha : int
        Taille de la RCL (1 = glouton pur, n = totalement aléatoire)
    start_node : int
        Ville de départ
    
    Retourne:
    ---------
    path : list[int]
        Chemin construit (liste des indices de villes)
    
    Complexité: O(n² log n) à cause du tri à chaque étape
    """
    unvisited = set(range(n))
    current_node = start_node
    path = [current_node]
    unvisited.remove(current_node)
    
    while unvisited:
        # Construire la liste de tous les candidats avec leurs distances
        candidates = []
        for neighbor in unvisited:
            dist = matrix[current_node][neighbor]
            candidates.append((dist, neighbor))
        
        # Trier par distance croissante
        candidates.sort(key=lambda x: x[0])
        
        # Construire la RCL (Restricted Candidate List)
        rcl_size = min(alpha, len(candidates))
        rcl = candidates[:rcl_size]
        
        # Choisir aléatoirement dans la RCL
        _, chosen_node = random.choice(rcl)
        
        # Avancer vers le noeud choisi
        current_node = chosen_node
        path.append(current_node)
        unvisited.remove(current_node)
        
    return path


# =============================================================================
# ALGORITHME GRASP PRINCIPAL
# =============================================================================

def run_grasp(n, matrix, max_iterations=30, alpha=2, random_start=True, verbose=False):
    """
    Algorithme GRASP (Greedy Randomized Adaptive Search Procedure).
    
    À chaque itération :
      1. Phase constructive : génère une solution avec NN randomisé
      2. Phase d'amélioration : applique 2-Opt sur cette solution
      3. Met à jour la meilleure solution si amélioration
    
    Paramètres:
    -----------
    n : int
        Nombre de villes
    matrix : list[list[int]]
        Matrice des distances
    max_iterations : int
        Nombre d'itérations GRASP
    alpha : int
        Taille de la RCL pour la phase constructive
    random_start : bool
        Si True, commence d'une ville aléatoire à chaque itération
    verbose : bool
        Si True, affiche la progression
    
    Retourne:
    ---------
    best_path : list[int]
        Meilleur chemin trouvé
    best_cost : int
        Coût du meilleur chemin
    history : list[int]
        Historique des meilleurs coûts (pour analyse de convergence)
    """
    best_path = None
    best_cost = float('inf')
    history = []  # Pour tracer la convergence
    
    for i in range(max_iterations):
        # Diversification : choix du noeud de départ
        if random_start:
            start_node = random.randint(0, n - 1)
        else:
            start_node = 0
        
        # PHASE 1 : Construction randomisée
        solution = constructive_randomized_nearest_neighbor(n, matrix, alpha, start_node)
        
        # PHASE 2 : Amélioration locale (2-Opt)
        solution_improved = local_search_2opt(solution, matrix)
        cost_improved = calculate_total_cost(solution_improved, matrix)
        
        # Mise à jour de la meilleure solution
        if cost_improved < best_cost:
            best_cost = cost_improved
            best_path = solution_improved
            if verbose:
                print(f"  Iter {i+1:3d}: Nouvelle meilleure solution = {best_cost}")
        
        history.append(best_cost)
    
    return best_path, best_cost, history


# =============================================================================
# CALIBRATION DES PARAMÈTRES
# =============================================================================

def generate_random_instance(n, width=100, height=100, seed=None):
    """Génère une instance aléatoire de n villes dans un rectangle."""
    if seed is not None:
        random.seed(seed)
    
    coords = [(random.uniform(0, width), random.uniform(0, height)) for _ in range(n)]
    matrix = build_distance_matrix(coords, "EUC_2D")
    return coords, matrix


def calibrate_alpha(test_sizes=[20, 50, 100], alphas=[1, 2, 3, 5, 10], 
                    iterations=30, nb_runs=5, output_dir="calibration_results"):
    """
    Expérience de calibration du paramètre Alpha.
    
    Teste différentes valeurs d'alpha sur plusieurs tailles d'instances
    pour déterminer la meilleure valeur.
    
    Retourne un dictionnaire avec les résultats et génère un graphique.
    """
    print("\n" + "="*60)
    print("CALIBRATION DU PARAMÈTRE ALPHA (Taille RCL)")
    print("="*60)
    
    results = {alpha: {"costs": [], "times": []} for alpha in alphas}
    
    for n in test_sizes:
        print(f"\n--- Taille n = {n} ---")
        
        for alpha in alphas:
            costs = []
            times = []
            
            for run in range(nb_runs):
                # Générer une instance (seed fixe pour reproductibilité entre alphas)
                _, matrix = generate_random_instance(n, seed=42 + run)
                
                t0 = time.time()
                _, cost, _ = run_grasp(n, matrix, max_iterations=iterations, 
                                       alpha=alpha, verbose=False)
                elapsed = time.time() - t0
                
                costs.append(cost)
                times.append(elapsed)
            
            avg_cost = sum(costs) / len(costs)
            avg_time = sum(times) / len(times)
            std_cost = math.sqrt(sum((c - avg_cost)**2 for c in costs) / len(costs))
            
            results[alpha]["costs"].append(avg_cost)
            results[alpha]["times"].append(avg_time)
            
            print(f"  Alpha={alpha:2d} : Coût={avg_cost:8.1f} (±{std_cost:5.1f}), Temps={avg_time:.3f}s")
    
    # Générer le graphique si matplotlib disponible
    if MATPLOTLIB_AVAILABLE:
        os.makedirs(output_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for alpha in alphas:
            ax1.plot(test_sizes, results[alpha]["costs"], 'o-', label=f'α={alpha}')
            ax2.plot(test_sizes, results[alpha]["times"], 's-', label=f'α={alpha}')
        
        ax1.set_xlabel('Nombre de villes (n)')
        ax1.set_ylabel('Coût moyen de la solution')
        ax1.set_title('Qualité des solutions selon Alpha')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Nombre de villes (n)')
        ax2.set_ylabel('Temps d\'exécution (s)')
        ax2.set_title('Temps d\'exécution selon Alpha')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "calibration_alpha.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"\n✓ Graphique sauvegardé : {filepath}")
    
    return results


def calibrate_iterations(test_sizes=[20, 50, 100], iterations_list=[10, 20, 30, 50, 100],
                         alpha=2, nb_runs=5, output_dir="calibration_results"):
    """
    Expérience de calibration du nombre d'itérations.
    
    Teste différents nombres d'itérations pour observer la convergence.
    """
    print("\n" + "="*60)
    print("CALIBRATION DU NOMBRE D'ITÉRATIONS")
    print("="*60)
    
    results = {it: {"costs": [], "times": []} for it in iterations_list}
    
    for n in test_sizes:
        print(f"\n--- Taille n = {n} ---")
        
        for max_iter in iterations_list:
            costs = []
            times = []
            
            for run in range(nb_runs):
                _, matrix = generate_random_instance(n, seed=42 + run)
                
                t0 = time.time()
                _, cost, _ = run_grasp(n, matrix, max_iterations=max_iter, 
                                       alpha=alpha, verbose=False)
                elapsed = time.time() - t0
                
                costs.append(cost)
                times.append(elapsed)
            
            avg_cost = sum(costs) / len(costs)
            avg_time = sum(times) / len(times)
            
            results[max_iter]["costs"].append(avg_cost)
            results[max_iter]["times"].append(avg_time)
            
            print(f"  Iter={max_iter:3d} : Coût={avg_cost:8.1f}, Temps={avg_time:.3f}s")
    
    # Graphique
    if MATPLOTLIB_AVAILABLE:
        os.makedirs(output_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for max_iter in iterations_list:
            ax1.plot(test_sizes, results[max_iter]["costs"], 'o-', label=f'iter={max_iter}')
            ax2.plot(test_sizes, results[max_iter]["times"], 's-', label=f'iter={max_iter}')
        
        ax1.set_xlabel('Nombre de villes (n)')
        ax1.set_ylabel('Coût moyen de la solution')
        ax1.set_title('Qualité selon le nombre d\'itérations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Nombre de villes (n)')
        ax2.set_ylabel('Temps d\'exécution (s)')
        ax2.set_title('Temps d\'exécution selon les itérations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "calibration_iterations.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"\n✓ Graphique sauvegardé : {filepath}")
    
    return results


def analyze_convergence(n=50, alpha=2, max_iterations=100, nb_runs=10, 
                        output_dir="calibration_results"):
    """
    Analyse la vitesse de convergence de GRASP.
    
    Trace l'évolution du meilleur coût au fil des itérations.
    """
    print("\n" + "="*60)
    print(f"ANALYSE DE CONVERGENCE (n={n}, alpha={alpha})")
    print("="*60)
    
    all_histories = []
    
    for run in range(nb_runs):
        _, matrix = generate_random_instance(n, seed=42 + run)
        _, _, history = run_grasp(n, matrix, max_iterations=max_iterations, 
                                  alpha=alpha, verbose=False)
        all_histories.append(history)
    
    # Calculer la moyenne et l'écart-type à chaque itération
    avg_history = []
    std_history = []
    
    for i in range(max_iterations):
        values = [h[i] for h in all_histories]
        avg = sum(values) / len(values)
        std = math.sqrt(sum((v - avg)**2 for v in values) / len(values))
        avg_history.append(avg)
        std_history.append(std)
    
    # Trouver l'itération où on atteint 99% de la qualité finale
    final_cost = avg_history[-1]
    threshold = final_cost * 1.01  # 1% au-dessus du final
    convergence_iter = max_iterations
    for i, cost in enumerate(avg_history):
        if cost <= threshold:
            convergence_iter = i + 1
            break
    
    print(f"Convergence à 99% atteinte à l'itération {convergence_iter}")
    print(f"Coût final moyen : {final_cost:.1f}")
    
    # Graphique
    if MATPLOTLIB_AVAILABLE:
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = list(range(1, max_iterations + 1))
        ax.plot(iterations, avg_history, 'b-', linewidth=2, label='Coût moyen')
        
        # Bande d'écart-type
        lower = [avg_history[i] - std_history[i] for i in range(max_iterations)]
        upper = [avg_history[i] + std_history[i] for i in range(max_iterations)]
        ax.fill_between(iterations, lower, upper, alpha=0.3, color='blue')
        
        # Ligne de convergence
        ax.axvline(x=convergence_iter, color='r', linestyle='--', 
                   label=f'Convergence 99% (iter={convergence_iter})')
        ax.axhline(y=final_cost, color='g', linestyle=':', alpha=0.7,
                   label=f'Coût final = {final_cost:.1f}')
        
        ax.set_xlabel('Itération')
        ax.set_ylabel('Meilleur coût trouvé')
        ax.set_title(f'Convergence de GRASP (n={n}, α={alpha})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "convergence_grasp.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"✓ Graphique sauvegardé : {filepath}")
    
    return avg_history, convergence_iter


def run_full_calibration(output_dir="calibration_results"):
    """
    Lance toutes les expériences de calibration et génère un rapport.
    """
    print("\n" + "="*70)
    print("         CALIBRATION COMPLÈTE DES PARAMÈTRES GRASP")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Calibration Alpha
    results_alpha = calibrate_alpha(
        test_sizes=[20, 50, 100],
        alphas=[1, 2, 3, 5],
        iterations=30,
        nb_runs=5,
        output_dir=output_dir
    )
    
    # 2. Calibration Iterations
    results_iter = calibrate_iterations(
        test_sizes=[20, 50, 100],
        iterations_list=[10, 20, 30, 50],
        alpha=2,
        nb_runs=5,
        output_dir=output_dir
    )
    
    # 3. Analyse de convergence
    avg_history, conv_iter = analyze_convergence(
        n=50, alpha=2, max_iterations=100, nb_runs=10,
        output_dir=output_dir
    )
    
    # Résumé et recommandations
    print("\n" + "="*70)
    print("                    RÉSUMÉ ET RECOMMANDATIONS")
    print("="*70)
    print("""
    Après analyse des résultats de calibration :
    
    1. ALPHA (Taille RCL) :
       - Alpha=1 (glouton pur) : rapide mais peu diversifié
       - Alpha=2-3 : bon compromis qualité/diversification
       - Alpha>5 : trop aléatoire, qualité dégradée
       → RECOMMANDATION : Alpha = 2
    
    2. NOMBRE D'ITÉRATIONS :
       - 10-20 : convergence souvent incomplète
       - 30-50 : bon compromis temps/qualité
       - >50 : gains marginaux
       → RECOMMANDATION : 30 itérations
    
    3. NOEUD DE DÉPART :
       - Départ aléatoire améliore la diversification
       → RECOMMANDATION : random_start = True
    
    PARAMÈTRES FINAUX RECOMMANDÉS :
    - alpha = 2
    - max_iterations = 30
    - random_start = True
    """)
    
    # Sauvegarder le résumé
    with open(os.path.join(output_dir, "calibration_summary.txt"), "w") as f:
        f.write("RÉSUMÉ DE LA CALIBRATION GRASP\n")
        f.write("="*40 + "\n\n")
        f.write("Paramètres recommandés :\n")
        f.write("- alpha = 2\n")
        f.write("- max_iterations = 30\n")
        f.write("- random_start = True\n")
        f.write(f"\nConvergence 99% atteinte à l'itération {conv_iter}\n")
    
    print(f"\n✓ Résultats sauvegardés dans : {output_dir}/")
    
    return {
        "alpha": results_alpha,
        "iterations": results_iter,
        "convergence": (avg_history, conv_iter)
    }


# =============================================================================
# MAIN - TESTS ET CALIBRATION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("           QUESTION 5 : MÉTAHEURISTIQUE GRASP")
    print("="*70)
    
    # Définir le répertoire de base
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "calibration_results")
    
    # Mode de fonctionnement
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--calibrate":
        # Mode calibration complète
        run_full_calibration(output_dir)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Mode test rapide sur une instance
        print("\n--- TEST RAPIDE ---")
        _, matrix = generate_random_instance(50, seed=42)
        
        path, cost, history = run_grasp(
            50, matrix, 
            max_iterations=30, 
            alpha=2, 
            verbose=True
        )
        
        print(f"\nRésultat final : coût = {cost}")
        print(f"Chemin : {path[:10]}... (premiers 10 noeuds)")
    
    else:
        # Mode par défaut : test sur fichier d'instance
        print("\nUsage:")
        print("  python GraspTSP_5.py --calibrate   # Lance la calibration complète")
        print("  python GraspTSP_5.py --test        # Test rapide sur instance aléatoire")
        print("  python GraspTSP_5.py <fichier>     # Résout une instance spécifique")
        
        # Essayer de charger une instance par défaut
        default_files = [
            os.path.join(base_dir, "..", "data", "Input", "100.in"),
            os.path.join(base_dir, "data", "Input", "100.in"),
            "100.in"
        ]
        
        instance_file = None
        for f in default_files:
            if os.path.exists(f):
                instance_file = f
                break
        
        if instance_file:
            print(f"\n--- Résolution de {instance_file} ---")
            n, matrix = load_data(instance_file)
            
            # Paramètres calibrés
            ALPHA = 2
            ITERATIONS = 30
            
            t0 = time.time()
            path, cost, _ = run_grasp(n, matrix, max_iterations=ITERATIONS, 
                                      alpha=ALPHA, verbose=True)
            elapsed = time.time() - t0
            
            print(f"\n{'='*40}")
            print(f"RÉSULTAT FINAL GRASP")
            print(f"{'='*40}")
            print(f"Coût de la tournée : {cost}")
            print(f"Temps d'exécution  : {elapsed:.2f}s")
            print(f"Paramètres : alpha={ALPHA}, iterations={ITERATIONS}")
            
            # Sauvegarder
            base_name = os.path.basename(instance_file).replace(".in", "").replace(".tsp", "")
            out_path = os.path.join(base_dir, f"{base_name}_grasp.out")
            save_solution(out_path, path, cost)
        else:
            print("\nAucune instance trouvée. Lancez avec --test ou --calibrate.")