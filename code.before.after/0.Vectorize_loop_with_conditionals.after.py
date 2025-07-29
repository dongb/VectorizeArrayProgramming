import numpy as np

def relu_clip_vectorized(X, T):
    return np.minimum(np.maximum(X, 0), T)

def conditional_mask_vectorized(A, r, c, threshold):
    val = A + r[:, np.newaxis] * c[np.newaxis, :]
    return (val > threshold).astype(int)

def main():
    # Set random seed for reproducible results
    np.random.seed(42)
    
    print("Testing Vectorized Functions")
    print("=" * 40)
    
    # Test data for relu_clip functions
    X = np.random.randn(5, 4) * 10  # Random matrix with values that can be negative or > T
    T = 5.0  # Threshold value
    
    # Run vectorized implementation
    result_vectorized = relu_clip_vectorized(X, T)
    
    print(f"ReLU Clip Vectorized Implementation")
    print(f"Input shape: {X.shape}")
    print(f"Threshold T: {T}")
    print()
    
    # Test data for conditional_mask functions
    A = np.random.randn(4, 3) * 2  # Base matrix
    r = np.random.randn(4)         # Row vector
    c = np.random.randn(3)         # Column vector
    threshold = 1.0
    
    # Run vectorized implementation
    mask_vectorized = conditional_mask_vectorized(A, r, c, threshold)
    
    print(f"Conditional Mask Vectorized Implementation")
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
    print(result_vectorized[:3, :3])
    print()
    
    print("Conditional Mask:")
    print("A matrix:")
    print(A)
    print("r vector:")
    print(r)
    print("c vector:")
    print(c)
    print("A + r*c (computed values):")
    val_computed = A + r[:, np.newaxis] * c[np.newaxis, :]
    print(val_computed)
    print("Mask result:")
    print(mask_vectorized)

if __name__ == "__main__":
    main() 