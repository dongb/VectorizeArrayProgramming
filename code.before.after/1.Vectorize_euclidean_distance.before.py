import numpy as np

def pairwise_distances_loop(A, B):
    m, n = A.shape[0], B.shape[0]
    D = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = np.linalg.norm(A[i] - B[j])
    return D

def main():
    # Create test data
    np.random.seed(42)  # For reproducible results
    A = np.random.rand(5, 3)  # 5 points in 3D space
    B = np.random.rand(4, 3)  # 4 points in 3D space
    
    print("Testing pairwise distance calculations (Loop-based implementation)...")
    print(f"Array A shape: {A.shape}")
    print(f"Array B shape: {B.shape}")
    print()
    
    # Run loop implementation
    print("Running loop-based implementation...")
    D_loop = pairwise_distances_loop(A, B)
    
    print(f"Result shape: {D_loop.shape}")
    print()
    
    print("Sample of distance matrix:")
    print(D_loop[:3, :3])  # Show first 3x3 portion
    
    # Performance test with larger arrays
    print("\nPerformance test with larger arrays...")
    A_large = np.random.rand(100, 10)
    B_large = np.random.rand(80, 10)
    
    import time
    
    # Time loop version
    start = time.time()
    D_loop_large = pairwise_distances_loop(A_large, B_large)
    loop_time = time.time() - start
    
    print(f"Loop implementation time: {loop_time:.4f} seconds")
    print(f"Large array result shape: {D_loop_large.shape}")

if __name__ == "__main__":
    main() 