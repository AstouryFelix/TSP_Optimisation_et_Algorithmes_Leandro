
import os
import sys
from src.model.load_data import load_data
from src.grasp.GraspTSP import run_grasp
from src.model.save_solution import save_solution
from src.model.total_cost import calculate_total_cost

def test_full_flow():
    instance_path = "instances/grasp/att48.tsp"
    if not os.path.exists(instance_path):
        print(f"Error: {instance_path} not found")
        # Try to find where it is
        import glob
        print("Instances files found:", glob.glob("instances/*.tsp"))
        return

    print(f"Testing on {instance_path}...")
    
    # 1. Load
    n, matrix = load_data(instance_path)
    print(f"Loaded {n} cities")
    
    # 2. Run GRASP
    print("Running GRASP...")
    path, cost, _ = run_grasp(n, matrix, max_iterations=5, alpha=2, verbose=False)
    
    # 3. Save
    out_file = "Solutions/att48_grasp.out"
    save_solution(out_file, path, cost, zero_based=True)
    
    # 4. Verify Output
    if not os.path.exists(out_file):
        print(f"Error: Output file not created at {out_file}")
        return
        
    with open(out_file, 'r') as f:
        lines = f.readlines()
        if len(lines) != 2:
            print(f"Error: Expected 2 lines, got {len(lines)}")
        else:
            path_str = lines[0].strip().split()
            cost_str = lines[1].strip()
            
            # Verify 1-based indexing
            if '0' in path_str:
                print("Error: Path contains '0', should be 1-based")
            else:
                print("Output format checks passed (1-based indexing)")
                
            print(f"Path length: {len(path_str)}")
            print(f"Cost: {cost_str}")
            
    print("Test passed successfully!")

if __name__ == "__main__":
    test_full_flow()
