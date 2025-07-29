import numpy as np

def relu_clip_loop(X, T):
    m, n = X.shape
    result = np.empty_like(X)
    for i in range(m):
        for j in range(n):
            x = X[i, j]
            if x < 0:
                result[i, j] = 0
            elif x > T:
                result[i, j] = T
            else:
                result[i, j] = x
    return result

def relu_clip_vectorized(X, T):
    return np.minimum(np.maximum(X, 0), T)

def conditional_mask_loop(A, r, c, threshold):
    m, n = A.shape
    result = np.zeros_like(A, dtype=int)
    for i in range(m):
        for j in range(n):
            val = A[i, j] + r[i] * c[j]
            if val > threshold:
                result[i, j] = 1
    return result

def conditional_mask_vectorized(A, r, c, threshold):
    val = A + r[:, np.newaxis] * c[np.newaxis, :]
    return (val > threshold).astype(int)

def main():
    # Set random seed for reproducible results
    np.random.seed(42)
    
    print("Testing ReLU Clip Functions")
    print("=" * 40)
    
    # Test data for relu_clip functions
    X = np.random.randn(5, 4) * 10  # Random matrix with values that can be negative or > T
    T = 5.0  # Threshold value
    
    # Run both implementations
    result_loop = relu_clip_loop(X, T)
    result_vectorized = relu_clip_vectorized(X, T)
    
    # Validate results are the same
    relu_match = np.allclose(result_loop, result_vectorized)
    print(f"ReLU Clip - Results match: {relu_match}")
    print(f"Input shape: {X.shape}")
    print(f"Threshold T: {T}")
    print()
    
    print("Testing Conditional Mask Functions")
    print("=" * 40)
    
    # Test data for conditional_mask functions
    A = np.random.randn(4, 3) * 2  # Base matrix
    r = np.random.randn(4)         # Row vector
    c = np.random.randn(3)         # Column vector
    threshold = 1.0
    
    # Run both implementations
    mask_loop = conditional_mask_loop(A, r, c, threshold)
    mask_vectorized = conditional_mask_vectorized(A, r, c, threshold)
    
    # Validate results are the same
    mask_match = np.array_equal(mask_loop, mask_vectorized)
    print(f"Conditional Mask - Results match: {mask_match}")
    print(f"Matrix A shape: {A.shape}")
    print(f"Vector r shape: {r.shape}")
    print(f"Vector c shape: {c.shape}")
    print(f"Threshold: {threshold}")
    print()
    
    # Show some example values for verification
    print("Sample Results")
    print("=" * 40)
    print("ReLU Clip (first 3x3):")
    print("Input:")
    print(X[:3, :3])
    print("Output (both should be identical):")
    print(result_loop[:3, :3])
    print()
    
    print("Conditional Mask:")
    print("A + r*c (computed values):")
    val_computed = A + r[:, np.newaxis] * c[np.newaxis, :]
    print(val_computed)
    print("Mask result (both should be identical):")
    print(mask_loop)
    print()
    
    # Overall validation
    all_tests_pass = relu_match and mask_match
    print(f"All tests pass: {all_tests_pass}")

if __name__ == "__main__":
    main()