def calculate_total_cost(path, matrix):
    """
    Calcule le coût total d'un cycle hamiltonien.
    
    Args:
        path: Liste des sommets dans l'ordre de visite (indices 0-based)
        matrix: Matrice des distances
        
    Returns:
        Coût total du cycle (incluant le retour au point de départ)
    """
    cost = 0
    n = len(path)
    for i in range(n):
        cost += matrix[path[i]][path[(i + 1) % n]]
    return cost

if __name__ == "__main__":
    # Test simple
    from src.model.load_data import load_data
    n, mat = load_data("data/Input/17.in")
    test_path = [0,3,12,6,7,16,5,13,14,2,10,9,1,4,8,11,15]
    print(f"Coût du chemin : {calculate_total_cost(test_path, mat)}")

