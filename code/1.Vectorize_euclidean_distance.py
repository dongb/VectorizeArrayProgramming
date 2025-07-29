import numpy as np

def pairwise_distances_loop(A, B):
    m, n = A.shape[0], B.shape[0]
    D = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = np.linalg.norm(A[i] - B[j])
    return D

def pairwise_distances_broadcast(A, B):
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # shape: (m, n, d)
    D = np.linalg.norm(diff, axis=2)  # shape: (m, n)
    return D

def main():
    # Create test data
    np.random.seed(42)  # For reproducible results
    A = np.random.rand(5, 3)  # 5 points in 3D space
    B = np.random.rand(4, 3)  # 4 points in 3D space
    
    print("Testing pairwise distance calculations...")
    print(f"Array A shape: {A.shape}")
    print(f"Array B shape: {B.shape}")
    print()
    
    # Run both implementations
    print("Running loop-based implementation...")
    D_loop = pairwise_distances_loop(A, B)
    
    print("Running broadcast-based implementation...")
    D_broadcast = pairwise_distances_broadcast(A, B)
    
    print(f"Result shape: {D_loop.shape}")
    print()
    
    # Validate results are the same
    are_equal = np.allclose(D_loop, D_broadcast, rtol=1e-10)
    print(f"Results are equal: {are_equal}")
    
    if are_equal:
        print("✓ Both implementations produce identical results!")
    else:
        print("✗ Results differ!")
        print("Max difference:", np.max(np.abs(D_loop - D_broadcast)))
    
    print()
    print("Sample of distance matrix:")
    print(D_loop[:3, :3])  # Show first 3x3 portion
    
    # Performance comparison with larger arrays
    print("\nPerformance comparison with larger arrays...")
    A_large = np.random.rand(100, 10)
    B_large = np.random.rand(80, 10)
    
    import time
    
    # Time loop version
    start = time.time()
    D_loop_large = pairwise_distances_loop(A_large, B_large)
    loop_time = time.time() - start
    
    # Time broadcast version
    start = time.time()
    D_broadcast_large = pairwise_distances_broadcast(A_large, B_large)
    broadcast_time = time.time() - start
    
    print(f"Loop implementation time: {loop_time:.4f} seconds")
    print(f"Broadcast implementation time: {broadcast_time:.4f} seconds")
    print(f"Speedup: {loop_time / broadcast_time:.2f}x")
    
    # Verify large arrays also produce same results
    large_equal = np.allclose(D_loop_large, D_broadcast_large, rtol=1e-10)
    print(f"Large array results equal: {large_equal}")

if __name__ == "__main__":
    main()

