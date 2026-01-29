import os
import numpy as np

# Import avec gestion des chemins relatifs
try:
    from src.model.mds_coordinates import export_matrix_solution_to_json
except ImportError:
    try:
        from .mds_coordinates import export_matrix_solution_to_json
    except ImportError:
        from mds_coordinates import export_matrix_solution_to_json

def export_to_json(filename_instance, matrix, path, cost, suffix="_solution"):
    """Exporte la solution en JSON pour visualizer.js."""
    
    # Conversion des types numpy si nécessaire
    if len(path) > 0 and type(path[0]) == np.int64:
        path = [int(x) for x in path]
    
    try:
        inst_name = os.path.basename(filename_instance).replace(".tsp","").replace(".in","")
        json_path = f"data/Solutions/{inst_name}{suffix}.json"
        
        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        export_matrix_solution_to_json(
            filename=filename_instance,
            distance_matrix=matrix,
            initial_path=[],
            initial_cost=0,
            optimized_path=path, 
            optimized_cost=cost,
            output_filename=json_path
        )
        print(f"JSON exporté : {json_path}")
    except Exception as e:
        print(f"Warning: Impossible d'exporter le JSON ({e})")