"""
J-Integral Computation: From Nested Loops to Vectorized Implementation

This module demonstrates two approaches to computing the J-integral in fracture mechanics:
1. Classic loop-based implementation
2. Vectorized implementation using NumPy

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
from dataclasses import dataclass
from typing import List, Union


@dataclass
class GaussPoint:
    """Represents a Gauss integration point"""
    state: str  # 'intact' or 'damaged'
    weight: float
    strain_energy: float
    stress: np.ndarray  # 2x2 stress tensor
    displacement_grad: np.ndarray  # 2x2 displacement gradient tensor
    normal: np.ndarray  # 2D normal vector


@dataclass
class Element:
    """Represents a finite element"""
    on_integration_path: bool
    gauss_points: List[GaussPoint]


def strain_energy(gp: GaussPoint) -> float:
    """Compute strain energy density at a Gauss point"""
    return gp.strain_energy


def stress_tensor(gp: GaussPoint) -> np.ndarray:
    """Get stress tensor at a Gauss point"""
    return gp.stress


def displacement_gradient(gp: GaussPoint) -> np.ndarray:
    """Get displacement gradient at a Gauss point"""
    return gp.displacement_grad


def normal_vector(gp: GaussPoint) -> np.ndarray:
    """Get normal vector at a Gauss point"""
    return gp.normal


def compute_j_integral_loops(elements: List[Element]) -> float:
    """
    Classic loop-based implementation of J-integral computation.
    
    Args:
        elements: List of finite elements
        
    Returns:
        J-integral value
    """
    j_total = 0.0
    
    for elem in elements:
        if not elem.on_integration_path:
            continue
            
        for gp in elem.gauss_points:
            if gp.state == 'damaged':
                continue
                
            w = strain_energy(gp)
            sigma = stress_tensor(gp)
            grad_u = displacement_gradient(gp)
            n = normal_vector(gp)

            delta_1j = np.array([1.0, 0.0])
            term1 = w * delta_1j
            term2 = sigma @ grad_u[:, 0]
            integrand = term1 - term2
            j_local = integrand @ n

            j_total += j_local * gp.weight
            
    return j_total


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
        Tuple of (elements_list, vectorized_data)
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
    
    # Create elements list for loop-based implementation
    elements = []
    for i in range(num_elements):
        elem_on_path = on_path_mask[i * gauss_per_elem]
        gauss_points = []
        
        for j in range(gauss_per_elem):
            idx = i * gauss_per_elem + j
            gp = GaussPoint(
                state=state[idx],
                weight=weights[idx],
                strain_energy=W[idx],
                stress=sigma[idx],
                displacement_grad=grad_u[idx],
                normal=n[idx]
            )
            gauss_points.append(gp)
            
        elem = Element(on_integration_path=elem_on_path, gauss_points=gauss_points)
        elements.append(elem)
    
    vectorized_data = (W, sigma, grad_u, n, weights, valid_mask)
    
    return elements, vectorized_data


def compare_implementations():
    """Compare the loop-based and vectorized implementations."""
    print("J-Integral Computation: Loop-based vs Vectorized Implementation")
    print("=" * 70)
    
    # Generate sample data
    elements, (W, sigma, grad_u, n, weights, valid_mask) = generate_sample_data()
    
    # Time the loop-based implementation
    import time
    
    start_time = time.time()
    j_loops = compute_j_integral_loops(elements)
    loop_time = time.time() - start_time
    
    # Time the vectorized implementation
    start_time = time.time()
    j_vectorized = compute_j_integral_vectorized(W, sigma, grad_u, n, weights, valid_mask)
    vectorized_time = time.time() - start_time
    
    print(f"Loop-based J-integral:     {j_loops:.6f}")
    print(f"Vectorized J-integral:     {j_vectorized:.6f}")
    print(f"Difference:                {abs(j_loops - j_vectorized):.6e}")
    print()
    print(f"Loop-based time:           {loop_time:.6f} seconds")
    print(f"Vectorized time:           {vectorized_time:.6f} seconds")
    print(f"Speedup:                   {loop_time / vectorized_time:.2f}x")
    
    return j_loops, j_vectorized


def demonstrate_vectorized_only():
    """Demonstrate the standalone vectorized implementation."""
    print("\nStandalone Vectorized Implementation")
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


def realistic_example():
    """
    A more realistic example showing how this would work with actual FE data.
    This demonstrates the required data structure for real applications.
    """
    print("\n" + "="*60)
    print("REALISTIC APPLICATION EXAMPLE")
    print("="*60)
    
    # In a real application, you would have:
    print("In real applications, you need the following data:")
    print("1. Finite element mesh and integration path definition")
    print("2. Stress field σ_ij (from FE analysis)")
    print("3. Displacement gradient field ∂u_i/∂x_j (from FE analysis)")
    print("4. Strain energy density W (from constitutive relations)")
    print("5. Normal vectors n_j of integration path (from geometry)")
    
    print("\nLimitations of this example:")
    print("❌ Uses random data instead of real FE results")
    print("❌ Fixed normal vector [1,0] instead of actual geometry")
    print("❌ Simplified material state determination")
    print("❌ Missing actual path integral geometry")
    
    print("\nExtensions needed for real applications:")
    print("✅ Integration with FE solvers (e.g., FEniCS, ABAQUS)")
    print("✅ Geometric path definition and normal vector computation")
    print("✅ Complex material constitutive relations")
    print("✅ Adaptive integration path selection")
    
    # Demonstrate the computational structure that would be needed
    print(f"\nComputational structure demonstration (using simulated data):")
    
    # This shows the data pipeline structure
    class RealApplicationPipeline:
        def __init__(self):
            self.mesh = "FE_Mesh_Object"
            self.crack_tip = np.array([0.0, 0.0])
            self.integration_radius = 0.1
            
        def extract_integration_path(self):
            """Extract elements and Gauss points along integration contour"""
            return "path_elements", "path_gauss_points"
            
        def compute_stress_field(self):
            """From FE solution"""
            return "stress_tensor_field"
            
        def compute_displacement_gradients(self):
            """From FE solution"""
            return "displacement_gradient_field"
            
        def compute_normal_vectors(self):
            """From path geometry"""
            return "normal_vector_field"
    
    pipeline = RealApplicationPipeline()
    print(f"Data pipeline structure: {type(pipeline).__name__}")
    print("- Mesh: ", pipeline.mesh)
    print("- Crack tip: ", pipeline.crack_tip)
    print("- Integration radius: ", pipeline.integration_radius)


if __name__ == "__main__":
    # Run comparison between implementations
    compare_implementations()
    
    # Demonstrate standalone vectorized implementation
    demonstrate_vectorized_only()
    
    # Show realistic application considerations
    realistic_example() 