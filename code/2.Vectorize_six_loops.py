import numpy as np
import time

def tensor_six_naive(X):
    N = X.shape[0]
    T = np.zeros((N, N, N, N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    for m in range(N):
                        for n in range(N):
                            T[i,j,k,l,m,n] = (
                                np.sum((X[i] - X[j])**2) +
                                np.sum((X[k] - X[l])**2) +
                                np.sum((X[m] - X[n])**2)
                            )
    return T

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
    
    print(f"Testing with {N} points in {D}D space")
    print(f"Input shape: {X.shape}")
    print(f"Output tensor shape will be: ({N}, {N}, {N}, {N}, {N}, {N})")
    
    # Test naive implementation
    print("\nRunning naive implementation...")
    start_time = time.time()
    result_naive = tensor_six_naive(X)
    naive_time = time.time() - start_time
    print(f"Naive implementation took: {naive_time:.4f} seconds")
    
    # Test vectorized implementation
    print("\nRunning vectorized implementation...")
    start_time = time.time()
    result_vectorized = tensor_six_vectorized(X)
    vectorized_time = time.time() - start_time
    print(f"Vectorized implementation took: {vectorized_time:.4f} seconds")
    
    # Validate results are the same
    print("\nValidating results...")
    are_equal = np.allclose(result_naive, result_vectorized, rtol=1e-10, atol=1e-10)
    
    if are_equal:
        print("✅ SUCCESS: Both implementations produce identical results!")
    else:
        print("❌ ERROR: Results differ between implementations")
        max_diff = np.max(np.abs(result_naive - result_vectorized))
        print(f"Maximum absolute difference: {max_diff}")
    
    # Performance comparison
    if naive_time > 0 and vectorized_time > 0:
        speedup = naive_time / vectorized_time
        print(f"\nPerformance comparison:")
        print(f"Speedup: {speedup:.2f}x faster with vectorization")
    
    # Show some sample values
    print(f"\nSample tensor values:")
    print(f"T[0,0,0,0,0,0] = {result_vectorized[0,0,0,0,0,0]:.6f}")
    print(f"T[1,1,1,1,1,1] = {result_vectorized[1,1,1,1,1,1]:.6f}")
    print(f"T[0,1,2,3,0,1] = {result_vectorized[0,1,2,3,0,1]:.6f}")

if __name__ == "__main__":
    main()
