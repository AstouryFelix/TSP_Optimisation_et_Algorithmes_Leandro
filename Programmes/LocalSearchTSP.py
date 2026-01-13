import time
import random

def read_instance(filename):
    """
    Lit le fichier d'instance TSP au format spécifié.
    
    Structure du fichier :
    - Ligne 1 : Nombre de villes n
    - Lignes suivantes : La matrice de distance nxn (chaque ligne contient les distances vers les autres villes)
    
    Args:
        filename (str): Le chemin vers le fichier .in
        
    Returns:
        n (int): Nombre de villes
        matrix (list of list): Matrice des distances (symétrique pour un TSP symétrique)
    """
    with open(filename, 'r') as f:
        # Lecture de toutes les lignes du fichier
        lines = f.readlines()
        
        # Le premier nombre est le nombre de villes 'n'
        # .strip() nettoie les espaces/sauts de ligne
        n = int(lines[0].strip())
        
        matrix = []
        # On parcourt les lignes suivantes pour construire la matrice
        # range(1, n+1) car la ligne 0 est 'n', les n lignes suivantes sont la matrice
        for i in range(1, n + 1):
            # Transformation de la ligne de texte en liste d'entiers :
            # 1. lines[i].strip() : enlève les espaces aux extrémités
            # 2. .split() : découpe la chaîne par défaut sur les espaces "10 20 30" -> ["10", "20", "30"]
            # 3. map(int, ...) : convertit chaque chaîne en entier
            # 4. list(...) : transforme l'objet map en une vraie liste Python
            row = list(map(int, lines[i].strip().split()))
            matrix.append(row)
            
    return n, matrix

def calculate_total_cost(path, matrix):
    """Calcule le coût total d'un cycle"""
    cost = 0
    n = len(path)
    for i in range(n):
        u = path[i]
        v = path[(i + 1) % n] # Le modulo permet de relier le dernier au premier
        cost += matrix[u][v]
    return cost

# --- QUESTION 3 : HEURISTIQUE CONSTRUCTIVE (Nearest Neighbor) ---
def constructive_nearest_neighbor(n, matrix, start_node=0):
    """
    Implémente l'algorithme glouton "Heuristique du plus proche voisin".
    
    Principe :
    1. On part d'une ville de départ.
    2. On cherche la ville la plus proche non encore visitée.
    3. On s'y déplace et on recommence jusqu'à avoir tout visité.
    4. On retourne à la ville de départ (implicite dans le calcul du coût, 
       ici on retourne juste l'ordre des villes).
    
    Args:
        n (int): Nombre de villes
        matrix (list): Matrice des distances
        start_node (int): Ville de départ (par défaut 0)
        
    Returns:
        list: Chemin (ordre de visite des sommets, ex: [0, 5, 2, ...])
    """
    # Ensemble des villes non visitées pour une recherche rapide (O(1))
    unvisited = set(range(n))
    
    current_node = start_node
    path = [current_node] # On commence le chemin
    unvisited.remove(current_node) # On marque la ville de départ comme visitée
    
    # Tant qu'il reste des villes à visiter
    while unvisited:
        nearest_node = None
        min_dist = float('inf') # Infini positif pour commencer
        
        # On regarde toutes les villes candidates (non visitées)
        for neighbor in unvisited:
            dist = matrix[current_node][neighbor]
            
            # Si on trouve un voisin plus proche que le meilleur actuel
            if dist < min_dist:
                min_dist = dist
                nearest_node = neighbor
        
        # Une fois le tour des voisins fini, on valide le déplacement
        current_node = nearest_node
        path.append(current_node)     # Ajout au chemin
        unvisited.remove(current_node) # Marquage comme visité
        
    return path

# --- QUESTION 4 : RECHERCHE LOCALE (2-OPT) ---
def local_search_2opt(path, matrix):
    """
    Améliore un chemin existant en utilisant l'opérateur "2-opt".
    
    Technique :
    On supprime 2 arêtes non adjacentes et on reconnecte les chemins différemment
    si cela réduit la distance totale. Cela revient à inverser une sous-section du chemin.
    
    Stratégie "First Improvement" :
    Dès qu'on trouve une amélioration, on l'applique et on recommence la boucle.
    (Alternative : "Best Improvement" où on teste tout avant de choisir la meilleure).
    
    Args:
        path (list): Le chemin initial (ex: [0, 5, 2...])
        matrix (list): Matrice des distances
        
    Returns:
        list: Le chemin optimisé
    """
    n = len(path)
    improved = True
    best_path = path[:] # On travaille sur une copie pour ne pas modifier l'original par erreur
    
    # On continue tant qu'on arrive à améliorer le chemin
    while improved:
        improved = False
        
        # Double boucle pour tester toutes les paires d'arêtes (i, i+1) et (j, j+1)
        # L'arête 1 part de i vers i+1
        for i in range(n - 1):
            # L'arête 2 part de j vers j+1
            # On commence à i+2 pour ne pas prendre une arête adjacente à la première
            # (car échanger des arêtes adjacentes ne change rien en 2-opt basique)
            for j in range(i + 2, n): 
                
                # Cas particulier pour la fermeture du cycle :
                # Si j est la dernière ville (n-1), l'arête j->j+1 est (n-1)->0.
                # Si i est 0, l'arête i->i+1 est 0->1.
                # Ces deux arêtes sont adjacentes au sommet 0, donc on les ignore.
                if j == n - 1 and i == 0:
                    continue
                
                # Identification des sommets
                # Arête 1 : u -> v
                u = best_path[i]
                v = best_path[i+1]
                
                # Arête 2 : x -> y
                x = best_path[j]
                y = best_path[(j + 1) % n] # Modulo n pour boucler sur le premier élément si j est le dernier
                
                # Calcul des coûts
                # Coût actuel : dist(u,v) + dist(x,y)
                current_cost = matrix[u][v] + matrix[x][y]
                
                # Coût potentiel après échange : dist(u,x) + dist(v,y)
                # On connecte le début de la première arête au début de la seconde
                # et la fin de la première à la fin de la seconde (après inversion du segment)
                new_cost = matrix[u][x] + matrix[v][y]
                
                if new_cost < current_cost:
                    # AMÉLIORATION TROUVÉE !
                    
                    # On applique le mouvement 2-opt :
                    # Cela consiste à inverser (reverse) le segment de villes entre v et x inclus.
                    # En python : slice [i+1 : j+1] contient les villes de v à x.
                    # [::-1] inverse cette liste.
                    best_path[i+1 : j+1] = best_path[i+1 : j+1][::-1]
                    
                    improved = True
                    # On a trouvé une amélioration, on continue la boucle qui est controlée par 'improved'.
                    # Note : Dans une "First Improvement", on pourrait faire un 'break' ici pour
                    # redémarrer les boucles 'for' depuis le début (i=0) avec le nouveau chemin.
                    # Ici, on continue l'itération courante mais on est sûr de refaire un tour de 'while'.
        
    return best_path

# --- MAIN ---
if __name__ == "__main__":
    filename = "../data/Input/100.in" 
    
    try:
        print(f"--- Lecture de {filename} ---")
        n, matrix = read_instance(filename)
        print(f"Nombre de villes : {n}")
        
        # 1. Solution Initiale (Constructive)
        start_time = time.time()
        initial_path = constructive_nearest_neighbor(n, matrix, start_node=0)
        initial_cost = calculate_total_cost(initial_path, matrix)
        duration_init = time.time() - start_time
        
        print(f"\n[Constructive] Nearest Neighbor :")
        print(f"Coût : {initial_cost}")
        print(f"Temps : {duration_init:.4f} sec")
        
        # 2. Amélioration (Recherche Locale)
        start_time = time.time()
        optimized_path = local_search_2opt(initial_path, matrix)
        optimized_cost = calculate_total_cost(optimized_path, matrix)
        duration_opt = time.time() - start_time
        
        print(f"\n[Local Search] 2-Opt :")
        print(f"Coût : {optimized_cost}")
        print(f"Temps : {duration_opt:.4f} sec")
        print(f"Amélioration : {initial_cost - optimized_cost} points")
        
        # Vérification formelle du résultat
        print("\nSolution finale (Ordre des villes) :")
        print(optimized_path)

        # Génération du fichier de sortie demandé (format instance_method.out)
        # Nom : 100_local_search.out
        output_filename = "../data/Solutions/100_local_search.out"
        with open(output_filename, "w") as f_out:
            # Ligne 1 : Les villes séparées par un espace
            f_out.write(" ".join(map(str, optimized_path)) + "\n")
            # Ligne 2 : Le coût
            f_out.write(str(optimized_cost) + "\n")
        print(f"\nFichier '{output_filename}' généré avec succès.")

        # Génération du fichier JSON pour visualisation web (avec MDS)
        try:
            import sys
            import os
            
            # Ajout du dossier Tools au path pour l'import
            tools_path = os.path.join(os.path.dirname(__file__), 'Tools')
            if tools_path not in sys.path:
                sys.path.append(tools_path)
                
            from mds_coordinates import export_matrix_solution_to_json
            
            print("\n" + "="*60)
            print("📊 Génération du fichier JSON pour visualisation web")
            print("="*60)
            
            export_matrix_solution_to_json(
                filename=filename,
                distance_matrix=matrix,
                initial_path=initial_path,
                initial_cost=initial_cost,
                optimized_path=optimized_path,
                optimized_cost=optimized_cost,
                output_filename="../data/Solutions/100_solution.json"
            )
        except ImportError:
            print("\n⚠️  Module 'mds_coordinates' non trouvé.")
            print("Pour générer le JSON avec visualisation, installez: pip install scikit-learn numpy")
        except Exception as e:
            print(f"\n⚠️  Erreur lors de la génération du JSON: {e}")

    except FileNotFoundError:
        print(f"Erreur : Le fichier {filename} est introuvable.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")