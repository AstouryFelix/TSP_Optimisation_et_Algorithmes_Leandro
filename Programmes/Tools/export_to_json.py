import os
import numpy as np
from Tools.mds_coordinates import export_matrix_solution_to_json

def export_to_json(filename_instance, matrix, path, cost, suffix="_solution"):

    if type(path[1]) == np.int64 :
        path = [int(x) for x in path]
    
    print( matrix,      )
    print( path,         )
    print( cost,         )

    """Exporte la solution en JSON pour visualizer.js."""
    try:
        inst_name = os.path.basename(filename_instance).replace(".tsp","").replace(".in","")
        json_path = f"../data/Solutions/{inst_name}{suffix}.json"
        
        export_matrix_solution_to_json(
            filename=filename_instance,
            distance_matrix=matrix,
            initial_path=[],
            initial_cost=0,
            optimized_path=path, 
            optimized_cost=cost,
            output_filename=json_path
        )
    except Exception as e:
        print(f"Warning: Impossible d'exporter le JSON ({e})")