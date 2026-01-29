"""
Fonction d'Export JSON pour Visualisation TSP
==============================================

Ce module contient une fonction réutilisable pour exporter les résultats
d'un algorithme TSP au format JSON, compatible avec l'application de visualisation web.

Usage:
    from export_tsp_json import export_solution_to_json
    
    export_solution_to_json(
        filename="instance.tsp",
        coords=[(36.49, 7.49), (57.06, 9.51), ...],
        initial_path=[0, 1, 2, ...],
        initial_cost=50000,
        optimized_path=[0, 92, 27, ...],
        optimized_cost=45000,
        edge_weight_type="GEO"
    )
"""

import json


def export_solution_to_json(filename, coords, initial_path, initial_cost, 
                            optimized_path, optimized_cost, edge_weight_type="EUC_2D",
                            output_filename=None):
    """
    Exporte une solution TSP au format JSON pour la visualisation web.
    
    Args:
        filename (str): Nom du fichier d'instance (ex: "ali535.tsp")
        coords (list): Liste des coordonnées [(lat1, lon1), (lat2, lon2), ...]
        initial_path (list): Chemin initial (ex: [0, 1, 2, ...])
        initial_cost (int): Coût de la solution initiale
        optimized_path (list): Chemin optimisé (ex: [0, 92, 27, ...])
        optimized_cost (int): Coût de la solution optimisée
        edge_weight_type (str): Type de distance ("GEO", "EUC_2D", etc.)
        output_filename (str, optional): Nom du fichier de sortie. 
                                        Par défaut: <instance>_solution.json
    
    Returns:
        str: Nom du fichier JSON généré
    
    Exemple:
        >>> coords = [(36.49, 7.49), (57.06, 9.51), (30.22, 48.14)]
        >>> initial = [0, 1, 2]
        >>> optimized = [0, 2, 1]
        >>> export_solution_to_json(
        ...     filename="test.tsp",
        ...     coords=coords,
        ...     initial_path=initial,
        ...     initial_cost=1000,
        ...     optimized_path=optimized,
        ...     optimized_cost=900,
        ...     edge_weight_type="GEO"
        ... )
        'test_solution.json'
    """
    # Déterminer le nom du fichier de sortie
    if output_filename is None:
        # Extraire le nom de l'instance (enlever le chemin et l'extension)
        instance_name = filename.split("/")[-1].replace("\\", "/").split("/")[-1]
        instance_name = instance_name.replace(".tsp", "").replace(".in", "")
        output_filename = f"{instance_name}_solution.json"
    
    # Créer la structure JSON
    json_data = {
        "instance": filename.split("/")[-1].replace("\\", "/").split("/")[-1].replace(".tsp", "").replace(".in", ""),
        "n_cities": len(coords),
        "coordinates": [[lat, lon] for lat, lon in coords],
        "initial_path": initial_path,
        "initial_cost": initial_cost,
        "optimized_path": optimized_path,
        "optimized_cost": optimized_cost,
        "improvement": initial_cost - optimized_cost,
        "edge_weight_type": edge_weight_type
    }
    
    # Écrire le fichier JSON
    with open(output_filename, "w", encoding="utf-8") as f_json:
        json.dump(json_data, f_json, indent=2, ensure_ascii=False)
    
    print(f"Fichier '{output_filename}' généré avec succès (pour visualisation web).")
    
    return output_filename


# ==================== EXEMPLE D'INTÉGRATION ====================
if __name__ == "__main__":
    """
    Exemple d'utilisation de la fonction d'export.
    
    À intégrer dans votre script principal comme ceci:
    """
    
    # Exemple de données (à remplacer par vos vraies données)
    example_coords = [
        (36.49, 7.49),
        (57.06, 9.51),
        (30.22, 48.14),
        (5.15, -3.56),
        (34.59, -106.37)
    ]
    
    example_initial_path = [0, 1, 2, 3, 4]
    example_initial_cost = 15000
    
    example_optimized_path = [0, 2, 4, 1, 3]
    example_optimized_cost = 12000
    
    # Appel de la fonction
    export_solution_to_json(
        filename="example.tsp",
        coords=example_coords,
        initial_path=example_initial_path,
        initial_cost=example_initial_cost,
        optimized_path=example_optimized_path,
        optimized_cost=example_optimized_cost,
        edge_weight_type="GEO"
    )