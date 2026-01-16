def calculate_total_cost(path, matrix):
    """Calcule le coût total d'un cycle."""
    cost = 0
    n = len(path)
    for i in range(n):
        cost += matrix[path[i]][path[(i + 1) % n]]
    return cost