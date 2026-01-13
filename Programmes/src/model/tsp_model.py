"""
Modèle TSP et Fonctions Utilitaires
===================================
Ce module contient les fonctions de lecture de données, de calcul de distance,
et d'export de solutions.
"""

import math
import sys
import os

# --- LECTURE DES DONNÉES ---

def read_instance_in(filename):
    """Lit un fichier au format simple .in (Matrice brute)."""
    with open(filename, 'r') as f:
        lines = f.readlines()
        n = int(lines[0].strip())
        matrix = []
        for i in range(1, n + 1):
            row = list(map(int, lines[i].strip().split()))
            matrix.append(row)
    return n, matrix

def read_instance_tsp(filename):
    """Lit un fichier au format TSPLIB .tsp (Coordonnées)."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n = 0
    edge_weight_type = "EUC_2D"
    coords = []
    in_coord_section = False
    
    for line in lines:
        line = line.strip()
        if line.startswith("DIMENSION"):
            n = int(line.split(":")[-1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[-1].strip()
        elif line == "NODE_COORD_SECTION":
            in_coord_section = True
            continue
        elif line == "EOF":
            break
        elif in_coord_section and line:
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
    
    return n, coords, edge_weight_type

def build_distance_matrix(coords, edge_weight_type):
    """Calcule la matrice des distances à partir de coordonnées."""
    n = len(coords)
    matrix = [[0] * n for _ in range(n)]
    
    # Définition de la fonction de distance locale
    def geo_dist(c1, c2):
        PI = 3.141592
        RRR = 6378.388
        lat1, lon1 = c1
        lat2, lon2 = c2
        
        deg1, min1 = int(lat1), lat1 - int(lat1)
        lat1_rad = PI * (deg1 + 5.0 * min1 / 3.0) / 180.0
        deg2, min2 = int(lon1), lon1 - int(lon1)
        lon1_rad = PI * (deg2 + 5.0 * min2 / 3.0) / 180.0
        
        deg1, min1 = int(lat2), lat2 - int(lat2)
        lat2_rad = PI * (deg1 + 5.0 * min1 / 3.0) / 180.0
        deg2, min2 = int(lon2), lon2 - int(lon2)
        lon2_rad = PI * (deg2 + 5.0 * min2 / 3.0) / 180.0
        
        q1 = math.cos(lon1_rad - lon2_rad)
        q2 = math.cos(lat1_rad - lat2_rad)
        q3 = math.cos(lat1_rad + lat2_rad)
        return int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)

    def euc_dist(c1, c2):
        return int(math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) + 0.5)

    dist_func = geo_dist if edge_weight_type == "GEO" else euc_dist

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_func(coords[i], coords[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix

def load_data(filename):
    """Fonction générique pour charger n'importe quel fichier (.in ou .tsp)."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Fichier introuvable: {filename}")

    if filename.endswith(".tsp"):
        n, coords, w_type = read_instance_tsp(filename)
        matrix = build_distance_matrix(coords, w_type)
        return n, matrix
    else:
        # Par défaut on suppose .in
        n, matrix = read_instance_in(filename)
        return n, matrix

def calculate_total_cost(path, matrix):
    """Calcule le coût total d'un cycle."""
    cost = 0
    n = len(path)
    for i in range(n):
        cost += matrix[path[i]][path[(i + 1) % n]]
    return cost

def save_solution(filename, path, cost):
    """Sauvegarde la solution dans data/Solutions."""
    # Assurer que le dossier existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(" ".join(map(str, path)) + "\n")
        f.write(str(cost) + "\n")
    print(f"Solution sauvegardée : {filename}")

def export_to_json(filename_instance, matrix, path, cost, suffix="_solution"):
    """Exporte la solution en JSON pour visualizer.js."""
    try:
        # Import dynamique pour éviter erreur si Tools manquant
        # On remonte de 2 niveaux (src/model -> src -> Programmes -> Root)
        # Mais ici le fichier est dans src/model, donc __file__/../../
        
        # Adaptation du path pour trouver Tools qui est dans Programmes/Tools
        # Structure actuelle: Programmes/src/model/tsp_model.py
        # Tools est dans Programmes/Tools
        
        current_dir = os.path.dirname(__file__)
        tools_path = os.path.abspath(os.path.join(current_dir, "../../Tools"))
        
        if tools_path not in sys.path:
            sys.path.append(tools_path)
            
        from mds_coordinates import export_matrix_solution_to_json
        
        inst_name = os.path.basename(filename_instance).replace(".tsp","").replace(".in","")
        json_path = f"../../data/Solutions/{inst_name}{suffix}.json"
        
        export_matrix_solution_to_json(
            filename=filename_instance,
            distance_matrix=matrix,
            initial_path=[], initial_cost=0,
            optimized_path=path, optimized_cost=cost,
            output_filename=json_path
        )
    except Exception as e:
        print(f"Warning: Impossible d'exporter le JSON ({e})")
