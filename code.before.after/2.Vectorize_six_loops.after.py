import numpy as np
import time

def pairwise_sq_dists(X):
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # shape (N, N, D)
    return np.sum(diff ** 2, axis=2)                  # shape (N, N)

def tensor_six_vectorized(X):
    A = pairwise_sq_dists(X)  # shape (N, N)

    term1 = A[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    term2 = A[np.newaxis, np.newaxis, :, :, np.newaxis, np.newaxis]
    term3 = A[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]

    T = term1 + term2 + term3  # shape (N, N, N, N, N, N)
    return T

def main():
    # Create test data - small size since tensor grows as N^6
    np.random.seed(42)  # For reproducible results
    N = 4  # Small size to keep computation manageable
    D = 3  # 3D points
    X = np.random.randn(N, D)
    
    print(f"Testing vectorized six-loop implementation with {N} points in {D}D space")
    print(f"Input shape: {X.shape}")
    print(f"Output tensor shape will be: ({N}, {N}, {N}, {N}, {N}, {N})")
    
    # Test vectorized implementation
    print("\nRunning vectorized implementation...")
    start_time = time.time()
    result_vectorized = tensor_six_vectorized(X)
    vectorized_time = time.time() - start_time
    print(f"Vectorized implementation took: {vectorized_time:.4f} seconds")
    
    # Show some sample values
    print(f"\nSample tensor values:")
    print(f"T[0,0,0,0,0,0] = {result_vectorized[0,0,0,0,0,0]:.6f}")
    print(f"T[1,1,1,1,1,1] = {result_vectorized[1,1,1,1,1,1]:.6f}")
    print(f"T[0,1,2,3,0,1] = {result_vectorized[0,1,2,3,0,1]:.6f}")
    
    print(f"\nVectorized implementation completed successfully!")

if __name__ == "__main__":
    main() 