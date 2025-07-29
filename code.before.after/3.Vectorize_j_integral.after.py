"""
J-Integral Computation: Vectorized Implementation

This module demonstrates the vectorized approach to computing the J-integral in fracture mechanics 
using NumPy operations.

The J-integral is a path-independent contour integral used to characterize the intensity 
of the stress and strain field near the tip of a crack.

J = ∫_Γ (W δ_{1j} - σ_{ij} ∂u_i/∂x_1) n_j ds

Where:
- W: strain energy density
- σ_{ij}: stress tensor
- ∂u_i/∂x_j: displacement gradient
- n_j: normal vector to the path
- Γ: integration path around the crack tip
"""

import numpy as np

def compute_j_integral_vectorized(W: np.ndarray, sigma: np.ndarray, grad_u: np.ndarray, 
                                 n: np.ndarray, weights: np.ndarray, valid_mask: np.ndarray) -> float:
    """
    Vectorized implementation of J-integral computation using NumPy.
    
    Args:
        W: Strain energy density array [N]
        sigma: Stress tensor array [N, 2, 2]
        grad_u: Displacement gradient array [N, 2, 2]
        n: Normal vector array [N, 2]
        weights: Integration weights [N]
        valid_mask: Boolean mask for valid integration points [N]
        
    Returns:
        J-integral value
    """
    # Compute integrand
    delta_1j = np.array([1.0, 0.0])
    term1 = W[:, np.newaxis] * delta_1j
    term2 = np.einsum('nij,nj->ni', sigma, grad_u[:, :, 0])
    integrand = term1 - term2
    j_contrib = np.einsum('ni,ni->n', integrand, n)
    
    # Final integration with masking
    j_total = np.sum(j_contrib[valid_mask] * weights[valid_mask])
    
    return j_total

def generate_sample_data(num_elements: int = 20, gauss_per_elem: int = 4):
    """
    Generate sample data for demonstration purposes.
    
    Args:
        num_elements: Number of finite elements
        gauss_per_elem: Number of Gauss points per element
        
    Returns:
        Tuple of vectorized data arrays
    """
    total_gps = num_elements * gauss_per_elem
    
    # Generate random data
    np.random.seed(42)  # For reproducible results
    
    # Boolean mask indicating whether each element is on the integration path
    on_path_mask = np.random.rand(num_elements) > 0.5
    on_path_mask = np.repeat(on_path_mask, gauss_per_elem)
    
    # Material states: 'intact' or 'damaged'
    state = np.random.choice(['intact', 'damaged'], size=total_gps, p=[0.8, 0.2])
    valid_mask = (state == 'intact') & on_path_mask
    
    # Physical quantities (simulated)
    W = np.random.uniform(1.0, 2.0, size=total_gps)
    sigma = np.random.rand(total_gps, 2, 2)
    grad_u = np.random.rand(total_gps, 2, 2)
    n = np.tile(np.array([1.0, 0.0]), (total_gps, 1))
    weights = np.ones(total_gps)
    
    return W, sigma, grad_u, n, weights, valid_mask

def demonstrate_vectorized():
    """Demonstrate the standalone vectorized implementation."""
    print("Standalone Vectorized Implementation")
    print("=" * 40)
    
    # Simulate data: N elements with M Gauss points each
    num_elements = 20
    gauss_per_elem = 4
    total_gps = num_elements * gauss_per_elem

    # Boolean mask indicating whether each element is on the integration path
    on_path_mask = np.random.rand(num_elements) > 0.5
    on_path_mask = np.repeat(on_path_mask, gauss_per_elem)

    # Material states: 'intact' or 'damaged'
    state = np.random.choice(['intact', 'damaged'], size=total_gps, p=[0.8, 0.2])
    valid_mask = (state == 'intact') & on_path_mask

    # Physical quantities (simulated)
    W = np.random.uniform(1.0, 2.0, size=total_gps)
    sigma = np.random.rand(total_gps, 2, 2)
    grad_u = np.random.rand(total_gps, 2, 2)
    n = np.tile(np.array([1.0, 0.0]), (total_gps, 1))

    # Compute integrand
    delta_1j = np.array([1.0, 0.0])
    term1 = W[:, np.newaxis] * delta_1j
    term2 = np.einsum('nij,nj->ni', sigma, grad_u[:, :, 0])
    integrand = term1 - term2
    j_contrib = np.einsum('ni,ni->n', integrand, n)
    weights = np.ones_like(W)

    # Final integration
    j_total = np.sum(j_contrib[valid_mask] * weights[valid_mask])
    
    print(f"Vectorized J-integral = {j_total:.6f}")
    print(f"Number of valid integration points: {np.sum(valid_mask)}/{total_gps}")

def main():
    """Test the vectorized J-integral implementation."""
    print("J-Integral Computation: Vectorized Implementation")
    print("=" * 70)
    
    # Generate sample data
    W, sigma, grad_u, n, weights, valid_mask = generate_sample_data()
    
    # Time the vectorized implementation
    import time
    
    start_time = time.time()
    j_vectorized = compute_j_integral_vectorized(W, sigma, grad_u, n, weights, valid_mask)
    vectorized_time = time.time() - start_time
    
    print(f"Vectorized J-integral:     {j_vectorized:.6f}")
    print(f"Vectorized time:           {vectorized_time:.6f} seconds")
    print(f"Number of valid integration points: {np.sum(valid_mask)}/{len(W)}")
    
    # Demonstrate standalone vectorized implementation
    demonstrate_vectorized()
    
    print("\nVectorized implementation completed successfully!")

if __name__ == "__main__":
    main() 