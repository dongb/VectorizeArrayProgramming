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

def main():
    # Create test data - small size since tensor grows as N^6
    np.random.seed(42)  # For reproducible results
    N = 4  # Small size to keep computation manageable
    D = 3  # 3D points
    X = np.random.randn(N, D)
    
    print(f"Testing naive six-loop implementation with {N} points in {D}D space")
    print(f"Input shape: {X.shape}")
    print(f"Output tensor shape will be: ({N}, {N}, {N}, {N}, {N}, {N})")
    
    # Test naive implementation
    print("\nRunning naive implementation...")
    start_time = time.time()
    result_naive = tensor_six_naive(X)
    naive_time = time.time() - start_time
    print(f"Naive implementation took: {naive_time:.4f} seconds")
    
    # Show some sample values
    print(f"\nSample tensor values:")
    print(f"T[0,0,0,0,0,0] = {result_naive[0,0,0,0,0,0]:.6f}")
    print(f"T[1,1,1,1,1,1] = {result_naive[1,1,1,1,1,1]:.6f}")
    print(f"T[0,1,2,3,0,1] = {result_naive[0,1,2,3,0,1]:.6f}")
    
    print(f"\nNaive implementation completed successfully!")

if __name__ == "__main__":
    main() 