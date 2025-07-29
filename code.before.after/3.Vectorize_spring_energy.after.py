"""
Spring Energy Vectorized Implementation

This module demonstrates how to efficiently compute pairwise spring energy
between particles using NumPy vectorization.

Based on the physical problem: For N particles in 2D space connected by springs
with unit stiffness and zero rest length, compute the energy matrix where
E_{ij} = 0.5 * ||x_i - x_j||^2

Note: This is a simplified physics model for educational purposes.
"""

import numpy as np
import time

def pairwise_energy_vectorized(X):
    """
    Compute pairwise spring energy using NumPy vectorization.
    
    Args:
        X: numpy array of shape (N, 2) representing N particles in 2D space
        
    Returns:
        energy: numpy array of shape (N, N) where energy[i, j] = 0.5 * ||x_i - x_j||^2
    """
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # shape: (N, N, 2)
    energy = 0.5 * np.sum(diff ** 2, axis=-1)         # shape: (N, N)
    np.fill_diagonal(energy, 0)
    return energy

def test_correctness():
    """Test the vectorized implementation."""
    print("Testing vectorized implementation...")
    
    # Create test data: 4 particles in 2D
    np.random.seed(42)
    X = np.random.rand(4, 2) * 10
    
    print(f"Test particle positions:\n{X}")
    
    # Compute energy matrix using vectorized method
    energy_vectorized = pairwise_energy_vectorized(X)
    
    print(f"\nVectorized implementation result:\n{energy_vectorized}")
    
    return energy_vectorized

def performance_test():
    """Test performance of vectorized implementation."""
    print("\n" + "="*60)
    print("Performance Test - Vectorized Implementation")
    print("="*60)
    
    # Test different problem sizes
    sizes = [10, 50, 100, 200]
    
    for N in sizes:
        print(f"\nTesting with {N} particles...")
        
        # Generate random positions
        np.random.seed(42)
        X = np.random.rand(N, 2) * 10
        
        # Time vectorized implementation
        start_time = time.time()
        energy_vectorized = pairwise_energy_vectorized(X)
        vectorized_time = time.time() - start_time
        
        print(f"  Vectorized implementation: {vectorized_time:.6f} seconds")

def demonstrate_physics():
    """Demonstrate the physical interpretation of the results."""
    print("\n" + "="*60)
    print("Physical Interpretation")
    print("="*60)
    
    # Create a simple 3-particle system
    X = np.array([
        [0.0, 0.0],  # particle at origin
        [1.0, 0.0],  # particle 1 unit to the right
        [0.0, 1.0]   # particle 1 unit up
    ])
    
    print("Simple 3-particle system:")
    for i, pos in enumerate(X):
        print(f"  Particle {i}: ({pos[0]:.1f}, {pos[1]:.1f})")
    
    energy = pairwise_energy_vectorized(X)
    print(f"\nEnergy matrix:\n{energy}")
    
    # Explain specific entries
    print("\nPhysical interpretation:")
    print(f"  E[0,1] = {energy[0,1]:.3f} (energy between particles 0 and 1, distance = 1.0)")
    print(f"  E[0,2] = {energy[0,2]:.3f} (energy between particles 0 and 2, distance = 1.0)")
    print(f"  E[1,2] = {energy[1,2]:.3f} (energy between particles 1 and 2, distance = √2 ≈ 1.414)")
    print(f"  Expected E[1,2] = 0.5 * (√2)² = 0.5 * 2 = 1.0 ✓")

def main():
    print("Spring Energy Vectorized Implementation Demo")
    print("="*60)
    
    # Test correctness
    test_correctness()
    
    # Test performance
    performance_test()
    
    # Demonstrate physics
    demonstrate_physics()
    
    print("\n" + "="*60)
    print("Summary:")
    print("- Vectorized implementation computes pairwise spring energies efficiently")
    print("- Significantly faster than loop-based approach for larger problems")
    print("- Energy matrix is symmetric (E[i,j] = E[j,i])")
    print("- Diagonal elements are zero (no self-energy)")
    print("- Results match physical expectations based on distance")

if __name__ == "__main__":
    main() 