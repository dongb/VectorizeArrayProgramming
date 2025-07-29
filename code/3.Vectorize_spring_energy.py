"""
Spring Energy Vectorization Example

This module demonstrates how to efficiently compute pairwise spring energy
between particles using NumPy vectorization instead of nested loops.

Based on the physical problem: For N particles in 2D space connected by springs
with unit stiffness and zero rest length, compute the energy matrix where
E_{ij} = 0.5 * ||x_i - x_j||^2

Note: This is a simplified physics model for educational purposes.
"""

import numpy as np
import time


def pairwise_energy_naive(X):
    """
    Compute pairwise spring energy using nested loops (naive approach).
    
    Args:
        X: numpy array of shape (N, 2) representing N particles in 2D space
        
    Returns:
        energy: numpy array of shape (N, N) where energy[i, j] = 0.5 * ||x_i - x_j||^2
    """
    N = X.shape[0]
    energy = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                diff = X[i] - X[j]
                energy[i, j] = 0.5 * np.dot(diff, diff)
    return energy


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
    """Test that both implementations produce the same results."""
    print("Testing correctness of implementations...")
    
    # Create test data: 4 particles in 2D
    np.random.seed(42)
    X = np.random.rand(4, 2) * 10
    
    print(f"Test particle positions:\n{X}")
    
    # Compute energy matrices using both methods
    energy_naive = pairwise_energy_naive(X)
    energy_vectorized = pairwise_energy_vectorized(X)
    
    print(f"\nNaive implementation result:\n{energy_naive}")
    print(f"\nVectorized implementation result:\n{energy_vectorized}")
    
    # Check if results are close (within numerical precision)
    if np.allclose(energy_naive, energy_vectorized):
        print("✓ Both implementations produce identical results!")
    else:
        print("✗ Results differ!")
        print(f"Max difference: {np.max(np.abs(energy_naive - energy_vectorized))}")
    
    return energy_naive, energy_vectorized


def performance_comparison():
    """Compare performance of naive vs vectorized implementations."""
    print("\n" + "="*60)
    print("Performance Comparison")
    print("="*60)
    
    # Test different problem sizes
    sizes = [10, 50, 100, 200]
    
    for N in sizes:
        print(f"\nTesting with {N} particles...")
        
        # Generate random positions
        np.random.seed(42)
        X = np.random.rand(N, 2) * 10
        
        # Time naive implementation
        start_time = time.time()
        energy_naive = pairwise_energy_naive(X)
        naive_time = time.time() - start_time
        
        # Time vectorized implementation
        start_time = time.time()
        energy_vectorized = pairwise_energy_vectorized(X)
        vectorized_time = time.time() - start_time
        
        # Verify results are the same
        assert np.allclose(energy_naive, energy_vectorized), "Results don't match!"
        
        speedup = naive_time / vectorized_time if vectorized_time > 0 else float('inf')
        
        print(f"  Naive implementation:      {naive_time:.6f} seconds")
        print(f"  Vectorized implementation: {vectorized_time:.6f} seconds")
        print(f"  Speedup: {speedup:.2f}x")


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


if __name__ == "__main__":
    print("Spring Energy Vectorization Demo")
    print("="*60)
    
    # Test correctness
    test_correctness()
    
    # Compare performance
    performance_comparison()
    
    # Demonstrate physics
    demonstrate_physics()
    
    print("\n" + "="*60)
    print("Summary:")
    print("- Both implementations compute the same pairwise spring energies")
    print("- Vectorized approach is significantly faster for larger problems")
    print("- Energy matrix is symmetric (E[i,j] = E[j,i])")
    print("- Diagonal elements are zero (no self-energy)")
    print("- Results match physical expectations based on distance") 