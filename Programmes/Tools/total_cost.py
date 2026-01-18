try    : from Tools.load_data import *
except : from load_data import *

def calculate_total_cost(path, matrix):
    """Calcule le coût total d'un cycle."""
    cost = 0
    n = len(path)
    for i in range(n):
        cost += matrix[path[i]][path[(i + 1) % n]]
    return cost

if __name__ == "__main__":
    print(calculate_total_cost([0,3,12,6,7,16,5,13,14,2,10,9,1,4,8,11,15],load_data("../data/Input/17.in")[1]))
    print(calculate_total_cost([0,3,12,6,7,5,16,13,14,2,10,9,1,4,8,11,15],load_data("../data/Input/17.in")[1]))
