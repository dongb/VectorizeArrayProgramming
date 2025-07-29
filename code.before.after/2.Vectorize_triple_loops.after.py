import numpy as np
import time

def matrix_chain_order_vectorized(p):
    p = np.array(p)  # Convert to numpy array for vectorized indexing
    n = len(p) - 1
    m = np.full((n, n), np.inf)
    np.fill_diagonal(m, 0)

    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            k_vals = np.arange(i, j)
            left = m[i, k_vals]
            right = m[k_vals + 1, j]
            # Uses array indexing: p[k_vals + 1] - requires numpy arrays
            mult_cost = p[i] * p[k_vals + 1] * p[j + 1]
            total_cost = left + right + mult_cost
            m[i, j] = np.min(total_cost)
    
    return m

def matrix_chain_order_optimized(p):
    """
    Vectorized implementation with path recovery support.
    Returns both the cost matrix and the split point matrix.
    """
    p = np.array(p)  # Convert to numpy array for vectorized indexing
    n = len(p) - 1
    m = np.full((n, n), np.inf)
    s = np.full((n, n), -1, dtype=int)  # Split point matrix
    np.fill_diagonal(m, 0)

    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            k_vals = np.arange(i, j)
            left = m[i, k_vals]
            right = m[k_vals + 1, j]
            mult_cost = p[i] * p[k_vals + 1] * p[j + 1]
            total_cost = left + right + mult_cost
            best_k_idx = np.argmin(total_cost)
            m[i, j] = total_cost[best_k_idx]
            s[i, j] = k_vals[best_k_idx]  # Store the optimal split point
    
    return m, s

def build_optimal_parens(s, i, j):
    """
    Recursively build the optimal parenthesization string.
    """
    if i == j:
        return f"A{i+1}"
    else:
        k = s[i][j]
        left = build_optimal_parens(s, i, k)
        right = build_optimal_parens(s, k + 1, j)
        return f"({left} x {right})"

def main():
    """
    Test the vectorized matrix chain multiplication implementations.
    """
    print("Matrix Chain Order - Vectorized Implementations")
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
        
        # Run vectorized implementation  
        start_time = time.time()
        result_vectorized = matrix_chain_order_vectorized(p)
        vectorized_time = time.time() - start_time
        
        # Run optimized implementation with path recovery
        start_time = time.time()
        result_optimized, split_matrix = matrix_chain_order_optimized(p)
        optimized_time = time.time() - start_time
        
        print(f"Vectorized implementation time: {vectorized_time:.6f} seconds")
        print(f"Optimized implementation time:  {optimized_time:.6f} seconds")
        
        min_cost = result_vectorized[0, len(p) - 2]
        print(f"Minimum multiplication cost: {min_cost}")
        
        # Show optimal parenthesization
        optimal_parens = build_optimal_parens(split_matrix, 0, len(p) - 2)
        print(f"Optimal parenthesization: {optimal_parens}")
        
        # Show cost matrix for smaller examples
        if len(p) <= 7:
            print(f"\nCost matrix (upper triangular):")
            n = len(p) - 1
            for row in range(n):
                for col in range(n):
                    if col >= row:
                        print(f"{result_vectorized[row,col]:8.0f}", end=" ")
                    else:
                        print("       -", end=" ")
                print()
    
    print("\n" + "=" * 70)
    print("✅ Vectorized implementations completed!")
    
    # Demonstrate detailed example
    print(f"\nDetailed Example - Test Case 5:")
    p_example = [30, 35, 15, 5, 10, 20, 25]
    m_example, s_example = matrix_chain_order_optimized(p_example)
    
    print(f"Matrix dimensions: {p_example}")
    print(f"Matrices: ", end="")
    for i in range(len(p_example) - 1):
        print(f"A{i+1}({p_example[i]}x{p_example[i+1]})", end=" " if i < len(p_example) - 2 else "\n")
    
    print(f"Minimum cost: {m_example[0, len(p_example) - 2]}")
    print(f"Optimal parenthesization: {build_optimal_parens(s_example, 0, len(p_example) - 2)}")

if __name__ == "__main__":
    main() 