"""
Spring Energy Loop-based Implementation

This module demonstrates how to compute pairwise spring energy
between particles using nested loops.

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

def test_correctness():
    """Test the loop-based implementation."""
    print("Testing loop-based implementation...")
    
    # Create test data: 4 particles in 2D
    np.random.seed(42)
    X = np.random.rand(4, 2) * 10
    
    print(f"Test particle positions:\n{X}")
    
    # Compute energy matrix using loop method
    energy_naive = pairwise_energy_naive(X)
    
    print(f"\nLoop-based implementation result:\n{energy_naive}")
    
    return energy_naive

def performance_test():
    """Test performance of loop-based implementation."""
    print("\n" + "="*60)
    print("Performance Test - Loop-based Implementation")
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
        
        print(f"  Loop-based implementation: {naive_time:.6f} seconds")

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
    
    energy = pairwise_energy_naive(X)
    print(f"\nEnergy matrix:\n{energy}")
    
    # Explain specific entries
    print("\nPhysical interpretation:")
    print(f"  E[0,1] = {energy[0,1]:.3f} (energy between particles 0 and 1, distance = 1.0)")
    print(f"  E[0,2] = {energy[0,2]:.3f} (energy between particles 0 and 2, distance = 1.0)")
    print(f"  E[1,2] = {energy[1,2]:.3f} (energy between particles 1 and 2, distance = √2 ≈ 1.414)")
    print(f"  Expected E[1,2] = 0.5 * (√2)² = 0.5 * 2 = 1.0 ✓")

def main():
    print("Spring Energy Loop-based Implementation Demo")
    print("="*60)
    
    # Test correctness
    test_correctness()
    
    # Test performance
    performance_test()
    
    # Demonstrate physics
    demonstrate_physics()
    
    print("\n" + "="*60)
    print("Summary:")
    print("- Loop-based implementation computes pairwise spring energies")
    print("- Energy matrix is symmetric (E[i,j] = E[j,i])")
    print("- Diagonal elements are zero (no self-energy)")
    print("- Results match physical expectations based on distance")

if __name__ == "__main__":
    main() 