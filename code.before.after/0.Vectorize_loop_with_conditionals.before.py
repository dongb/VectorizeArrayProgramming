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

def conditional_mask_loop(A, r, c, threshold):
    m, n = A.shape
    result = np.zeros_like(A, dtype=int)
    for i in range(m):
        for j in range(n):
            val = A[i, j] + r[i] * c[j]
            if val > threshold:
                result[i, j] = 1
    return result

def main():
    # Set random seed for reproducible results
    np.random.seed(42)
    
    print("Testing Loop-based Functions")
    print("=" * 40)
    
    # Test data for relu_clip functions
    X = np.random.randn(5, 4) * 10  # Random matrix with values that can be negative or > T
    T = 5.0  # Threshold value
    
    # Run loop implementation
    result_loop = relu_clip_loop(X, T)
    
    print(f"ReLU Clip Loop Implementation")
    print(f"Input shape: {X.shape}")
    print(f"Threshold T: {T}")
    print()
    
    # Test data for conditional_mask functions
    A = np.random.randn(4, 3) * 2  # Base matrix
    r = np.random.randn(4)         # Row vector
    c = np.random.randn(3)         # Column vector
    threshold = 1.0
    
    # Run loop implementation
    mask_loop = conditional_mask_loop(A, r, c, threshold)
    
    print(f"Conditional Mask Loop Implementation")
    print(f"Matrix A shape: {A.shape}")
    print(f"Vector r shape: {r.shape}")
    print(f"Vector c shape: {c.shape}")
    print(f"Threshold: {threshold}")
    print()
    
    # Show some example values
    print("Sample Results")
    print("=" * 40)
    print("ReLU Clip (first 3x3):")
    print("Input:")
    print(X[:3, :3])
    print("Output:")
    print(result_loop[:3, :3])
    print()
    
    print("Conditional Mask:")
    print("A matrix:")
    print(A)
    print("r vector:")
    print(r)
    print("c vector:")
    print(c)
    print("Mask result:")
    print(mask_loop)

if __name__ == "__main__":
    main() 