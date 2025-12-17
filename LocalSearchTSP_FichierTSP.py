import time
import random
import math

def read_tsp_file(filename):
    """
    Lit un fichier au format TSPLIB (.tsp) avec des coordonnées géographiques.
    
    Structure du fichier :
    - En-tête avec métadonnées (NAME, TYPE, DIMENSION, EDGE_WEIGHT_TYPE, etc.)
    - Section NODE_COORD_SECTION avec les coordonnées de chaque ville
    - EOF pour marquer la fin
    
    Args:
        filename (str): Le chemin vers le fichier .tsp
        
    Returns:
        n (int): Nombre de villes
        coords (list of tuples): Liste des coordonnées [(lat1, lon1), (lat2, lon2), ...]
        edge_weight_type (str): Type de distance (GEO, EUC_2D, etc.)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Variables pour stocker les informations
    n = 0
    edge_weight_type = None
    coords = []
    in_coord_section = False
    
    for line in lines:
        line = line.strip()
        
        # Parse les métadonnées de l'en-tête
        if line.startswith("DIMENSION"):
            # Format: "DIMENSION: 535" ou "DIMENSION : 535"
            n = int(line.split(":")[-1].strip())
        
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            # Format: "EDGE_WEIGHT_TYPE: GEO"
            edge_weight_type = line.split(":")[-1].strip()
        
        # Détection du début de la section des coordonnées
        elif line == "NODE_COORD_SECTION":
            in_coord_section = True
            continue
        
        # Détection de la fin du fichier
        elif line == "EOF":
            break
        
        # Lecture des coordonnées
        elif in_coord_section and line:
            parts = line.split()
            if len(parts) >= 3:
                # Format: "1  36.49  7.49"
                # On ignore l'index (parts[0]) et on prend lat/lon
                node_id = int(parts[0])
                lat = float(parts[1])
                lon = float(parts[2])
                coords.append((lat, lon))
    
    return n, coords, edge_weight_type


def calculate_geo_distance(coord1, coord2):
    """
    Calcule la distance géographique entre deux points (format TSPLIB GEO).
    
    Formule utilisée par TSPLIB pour EDGE_WEIGHT_TYPE: GEO
    (approximation sphérique de la Terre)
    
    Args:
        coord1 (tuple): (latitude, longitude) du point 1
        coord2 (tuple): (latitude, longitude) du point 2
        
    Returns:
        int: Distance arrondie en kilomètres
    """
    PI = 3.141592
    RRR = 6378.388  # Rayon de la Terre en km (valeur TSPLIB)
    
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Conversion des degrés décimaux en radians (formule TSPLIB)
    deg1 = int(lat1)
    min1 = lat1 - deg1
    lat1_rad = PI * (deg1 + 5.0 * min1 / 3.0) / 180.0
    
    deg2 = int(lon1)
    min2 = lon1 - deg2
    lon1_rad = PI * (deg2 + 5.0 * min2 / 3.0) / 180.0
    
    deg1 = int(lat2)
    min1 = lat2 - deg1
    lat2_rad = PI * (deg1 + 5.0 * min1 / 3.0) / 180.0
    
    deg2 = int(lon2)
    min2 = lon2 - deg2
    lon2_rad = PI * (deg2 + 5.0 * min2 / 3.0) / 180.0
    
    # Formule de distance sphérique
    q1 = math.cos(lon1_rad - lon2_rad)
    q2 = math.cos(lat1_rad - lat2_rad)
    q3 = math.cos(lat1_rad + lat2_rad)
    
    distance = RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
    
    return int(distance)


def calculate_euclidean_distance(coord1, coord2):
    """
    Calcule la distance euclidienne 2D entre deux points.
    
    Args:
        coord1 (tuple): (x, y) du point 1
        coord2 (tuple): (x, y) du point 2
        
    Returns:
        int: Distance euclidienne arrondie
    """
    x1, y1 = coord1
    x2, y2 = coord2
    
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    return int(distance + 0.5)  # Arrondi standard TSPLIB


def build_distance_matrix(coords, edge_weight_type):
    """
    Construit la matrice de distances à partir des coordonnées.
    
    Args:
        coords (list): Liste des coordonnées
        edge_weight_type (str): Type de calcul ("GEO" ou "EUC_2D")
        
    Returns:
        matrix (list of list): Matrice de distances nxn
    """
    n = len(coords)
    matrix = [[0] * n for _ in range(n)]
    
    # Choisir la fonction de distance appropriée
    if edge_weight_type == "GEO":
        distance_func = calculate_geo_distance
    else:  # Par défaut, Euclidienne
        distance_func = calculate_euclidean_distance
    
    # Remplir la matrice (symétrique)
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_func(coords[i], coords[j])
            matrix[i][j] = dist
            matrix[j][i] = dist  # Symétrie
    
    return matrix


def calculate_total_cost(path, matrix):
    """Calcule le coût total d'un cycle"""
    cost = 0
    n = len(path)
    for i in range(n):
        u = path[i]
        v = path[(i + 1) % n]  # Le modulo permet de relier le dernier au premier
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
    path = [current_node]  # On commence le chemin
    unvisited.remove(current_node)  # On marque la ville de départ comme visitée
    
    # Tant qu'il reste des villes à visiter
    while unvisited:
        nearest_node = None
        min_dist = float('inf')  # Infini positif pour commencer
        
        # On regarde toutes les villes candidates (non visitées)
        for neighbor in unvisited:
            dist = matrix[current_node][neighbor]
            
            # Si on trouve un voisin plus proche que le meilleur actuel
            if dist < min_dist:
                min_dist = dist
                nearest_node = neighbor
        
        # Une fois le tour des voisins fini, on valide le déplacement
        current_node = nearest_node
        path.append(current_node)      # Ajout au chemin
        unvisited.remove(current_node)  # Marquage comme visité
        
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
    best_path = path[:]  # On travaille sur une copie pour ne pas modifier l'original par erreur
    
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
                y = best_path[(j + 1) % n]  # Modulo n pour boucler sur le premier élément si j est le dernier
                
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
    filename = "donnees_autre/ali535.tsp"  
    
    try:
        print(f"--- Lecture de {filename} ---")
        n, coords, edge_weight_type = read_tsp_file(filename)
        print(f"Nombre de villes : {n}")
        print(f"Type de distance : {edge_weight_type}")
        
        # Construction de la matrice de distances
        print("Construction de la matrice de distances...")
        matrix = build_distance_matrix(coords, edge_weight_type)
        
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

        # Génération du fichier de sortie
        output_filename = filename.split("/")[-1].replace(".tsp", "_local_search.out")
        with open(output_filename, "w") as f_out:
            # Ligne 1 : Les villes séparées par un espace
            f_out.write(" ".join(map(str, optimized_path)) + "\n")
            # Ligne 2 : Le coût
            f_out.write(str(optimized_cost) + "\n")
        print(f"\nFichier '{output_filename}' généré avec succès.")

    except FileNotFoundError:
        print(f"Erreur : Le fichier {filename} est introuvable.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
