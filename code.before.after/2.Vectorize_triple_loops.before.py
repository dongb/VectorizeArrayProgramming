import numpy as np
import time

def matrix_chain_order_naive(p):
    n = len(p) - 1
    m = np.zeros((n, n))
    
    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            m[i][j] = np.inf
            for k in range(i, j):
                # Uses scalar indexing: p[i], p[k+1], p[j+1] - works with Python lists
                cost = m[i][k] + m[k+1][j] + p[i] * p[k+1] * p[j+1]
                if cost < m[i][j]:
                    m[i][j] = cost
    return m

def main():
    """
    Test the naive loop-based matrix chain multiplication implementation.
    """
    print("Matrix Chain Order - Naive Loop-based Implementation")
    print("=" * 70)
    
    # Test cases with different matrix chain sizes
    test_cases = [
        [1, 2, 3, 4, 5],  # 4 matrices: 1x2, 2x3, 3x4, 4x5
        [5, 4, 6, 2, 7],  # 4 matrices: 5x4, 4x6, 6x2, 2x7
        [40, 20, 30, 10, 30],  # Example from algorithms textbook
        [1, 2, 3, 4, 5, 6, 7],  # 6 matrices
        [30, 35, 15, 5, 10, 20, 25]  # 6 matrices - classic example
    ]
    
    for i, p in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: Matrix dimensions {p}")
        print(f"Number of matrices: {len(p) - 1}")
        
        # Run naive implementation
        start_time = time.time()
        result_naive = matrix_chain_order_naive(p)
        naive_time = time.time() - start_time
        
        print(f"Naive implementation time: {naive_time:.6f} seconds")
        
        min_cost = result_naive[0, len(p) - 2]
        print(f"Minimum multiplication cost: {min_cost}")
        
        # Show cost matrix for smaller examples
        if len(p) <= 7:
            print(f"\nCost matrix (upper triangular):")
            n = len(p) - 1
            for row in range(n):
                for col in range(n):
                    if col >= row:
                        print(f"{result_naive[row,col]:8.0f}", end=" ")
                    else:
                        print("       -", end=" ")
                print()
    
    print("\n" + "=" * 70)
    print("✅ Naive loop-based implementation completed!")

if __name__ == "__main__":
    main() 