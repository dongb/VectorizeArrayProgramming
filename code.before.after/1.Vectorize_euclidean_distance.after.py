import numpy as np

def pairwise_distances_broadcast(A, B):
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # shape: (m, n, d)
    D = np.linalg.norm(diff, axis=2)  # shape: (m, n)
    return D

def main():
    # Create test data
    np.random.seed(42)  # For reproducible results
    A = np.random.rand(5, 3)  # 5 points in 3D space
    B = np.random.rand(4, 3)  # 4 points in 3D space
    
    print("Testing pairwise distance calculations (Vectorized implementation)...")
    print(f"Array A shape: {A.shape}")
    print(f"Array B shape: {B.shape}")
    print()
    
    # Run vectorized implementation
    print("Running broadcast-based implementation...")
    D_broadcast = pairwise_distances_broadcast(A, B)
    
    print(f"Result shape: {D_broadcast.shape}")
    print()
    
    print("Sample of distance matrix:")
    print(D_broadcast[:3, :3])  # Show first 3x3 portion
    
    # Performance test with larger arrays
    print("\nPerformance test with larger arrays...")
    A_large = np.random.rand(100, 10)
    B_large = np.random.rand(80, 10)
    
    import time
    
    # Time broadcast version
    start = time.time()
    D_broadcast_large = pairwise_distances_broadcast(A_large, B_large)
    broadcast_time = time.time() - start
    
    print(f"Broadcast implementation time: {broadcast_time:.4f} seconds")
    print(f"Large array result shape: {D_broadcast_large.shape}")

if __name__ == "__main__":
    main() 