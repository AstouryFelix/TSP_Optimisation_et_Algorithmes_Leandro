"""
Génération de Coordonnées 2D à partir d'une Matrice de Distances (MDS)
========================================================================

Ce module permet de générer des coordonnées 2D approximatives à partir d'une matrice
de distances TSP, permettant ainsi la visualisation des instances .in qui ne contiennent
pas de coordonnées explicites.

Utilise la technique MDS (Multidimensional Scaling) pour projeter les distances
en 2D tout en respectant au mieux les distances relatives.

Requirements:
    pip install scikit-learn numpy

Usage:
    from mds_coordinates import generate_coordinates_from_matrix
    
    coords = generate_coordinates_from_matrix(distance_matrix)
    # coords = [(x1, y1), (x2, y2), ...]
"""

import numpy as np
from sklearn.manifold import MDS


def generate_coordinates_from_matrix(distance_matrix):
    """
    Génère des coordonnées 2D à partir d'une matrice de distances en utilisant MDS.
    
    Le MDS (Multidimensional Scaling) projette les points dans un espace 2D de manière
    à préserver au mieux les distances entre les points.
    
    Args:
        distance_matrix (list of list or numpy.ndarray): Matrice de distances nxn
        
    Returns:
        list of tuples: Liste de coordonnées [(x1, y1), (x2, y2), ...]
        
    Exemple:
        >>> matrix = [
        ...     [0, 10, 15],
        ...     [10, 0, 20],
        ...     [15, 20, 0]
        ... ]
        >>> coords = generate_coordinates_from_matrix(matrix)
        >>> len(coords)
        3
    """
    # Convertir en numpy array si nécessaire
    dist_matrix = np.array(distance_matrix)
    
    # Créer le modèle MDS
    # metric=True : utilise les vraies distances (pas de transformation)
    # n_components=2 : projection en 2D
    # dissimilarity='precomputed' : on fournit directement les distances
    # random_state=42 : pour la reproductibilité
    mds = MDS(
        n_components=2,
        dissimilarity='precomputed',
        random_state=42,
        n_init=4,
        max_iter=300
    )
    
    # Calculer les coordonnées 2D
    coords_2d = mds.fit_transform(dist_matrix)
    
    # Convertir en liste de tuples (x, y)
    coordinates = [(float(x), float(y)) for x, y in coords_2d]
    
    return coordinates


def export_matrix_solution_to_json(filename, distance_matrix, initial_path, initial_cost,
                                   optimized_path, optimized_cost, output_filename=None):
    """
    Exporte une solution TSP (avec matrice de distances) au format JSON pour visualisation.
    
    Cette fonction génère automatiquement des coordonnées 2D à partir de la matrice
    de distances en utilisant MDS, puis exporte au format JSON.
    
    Args:
        filename (str): Nom du fichier d'instance (ex: "100.in")
        distance_matrix (list): Matrice de distances nxn
        initial_path (list): Chemin initial
        initial_cost (int): Coût initial
        optimized_path (list): Chemin optimisé
        optimized_cost (int): Coût optimisé
        output_filename (str, optional): Nom du fichier de sortie
        
    Returns:
        str: Nom du fichier JSON généré
        
    Exemple:
        >>> matrix = [[0, 10, 15], [10, 0, 20], [15, 20, 0]]
        >>> export_matrix_solution_to_json(
        ...     filename="test.in",
        ...     distance_matrix=matrix,
        ...     initial_path=[0, 1, 2],
        ...     initial_cost=45,
        ...     optimized_path=[0, 2, 1],
        ...     optimized_cost=35
        ... )
        'test_solution.json'
    """
    import json
    
    print("🔄 Génération des coordonnées 2D à partir de la matrice de distances...")
    
    # Générer les coordonnées avec MDS
    coordinates = generate_coordinates_from_matrix(distance_matrix)
    
    print(f"✅ Coordonnées générées pour {len(coordinates)} villes")
    
    # Déterminer le nom du fichier de sortie
    if output_filename is None:
        instance_name = filename.split("/")[-1].replace("\\", "/").split("/")[-1]
        instance_name = instance_name.replace(".in", "").replace(".tsp", "")
        output_filename = f"{instance_name}_solution.json"
    
    # Créer la structure JSON
    json_data = {
        "instance": filename.split("/")[-1].replace("\\", "/").split("/")[-1].replace(".in", "").replace(".tsp", ""),
        "n_cities": len(coordinates),
        "coordinates": coordinates,  # Coordonnées générées par MDS
        "initial_path": initial_path,
        "initial_cost": initial_cost,
        "optimized_path": optimized_path,
        "optimized_cost": optimized_cost,
        "improvement": initial_cost - optimized_cost,
        "edge_weight_type": "MATRIX_MDS",  # Indique que les coordonnées sont approximatives
        "note": "Coordonnées générées par MDS - approximation des positions réelles"
    }
    
    # Écrire le fichier JSON
    with open(output_filename, "w", encoding="utf-8") as f_json:
        json.dump(json_data, f_json, indent=2, ensure_ascii=False)
    
    print(f"✅ Fichier '{output_filename}' généré avec succès (pour visualisation web).")
    
    return output_filename


# ==================== EXEMPLE D'UTILISATION ====================
if __name__ == "__main__":
    """
    Exemple d'utilisation avec une petite matrice de test.
    """
    
    # Exemple de matrice de distances (3 villes)
    test_matrix = [
        [0,  10, 15],
        [10, 0,  20],
        [15, 20, 0]
    ]
    
    print("=== Test de génération de coordonnées MDS ===\n")
    
    # Générer les coordonnées
    coords = generate_coordinates_from_matrix(test_matrix)
    
    print("Coordonnées générées:")
    for i, (x, y) in enumerate(coords):
        print(f"  Ville {i}: ({x:.2f}, {y:.2f})")
    
    print("\n=== Test d'export JSON complet ===\n")
    
    # Export avec une solution fictive
    export_matrix_solution_to_json(
        filename="test_example.in",
        distance_matrix=test_matrix,
        initial_path=[0, 1, 2],
        initial_cost=45,  # 10 + 20 + 15
        optimized_path=[0, 2, 1],
        optimized_cost=35   # 15 + 20 + 10 (même distance, juste pour l'exemple)
    )
    
    print("\n📋 Pour l'intégrer dans votre code:")
    print("1. Assurez-vous d'avoir: pip install scikit-learn numpy")
    print("2. Importez: from mds_coordinates import export_matrix_solution_to_json")
    print("3. Appelez la fonction avec votre matrice de distances")
