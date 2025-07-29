"""
J-Integral Computation: Loop-based Implementation

This module demonstrates the classic loop-based approach to computing the J-integral in fracture mechanics.

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
from typing import List

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

def generate_sample_data(num_elements: int = 20, gauss_per_elem: int = 4):
    """
    Generate sample data for demonstration purposes.
    
    Args:
        num_elements: Number of finite elements
        gauss_per_elem: Number of Gauss points per element
        
    Returns:
        List of elements for loop-based implementation
    """
    total_gps = num_elements * gauss_per_elem
    
    # Generate random data
    np.random.seed(42)  # For reproducible results
    
    # Boolean mask indicating whether each element is on the integration path
    on_path_mask = np.random.rand(num_elements) > 0.5
    on_path_mask = np.repeat(on_path_mask, gauss_per_elem)
    
    # Material states: 'intact' or 'damaged'
    state = np.random.choice(['intact', 'damaged'], size=total_gps, p=[0.8, 0.2])
    
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
    
    return elements

def main():
    """Test the loop-based J-integral implementation."""
    print("J-Integral Computation: Loop-based Implementation")
    print("=" * 70)
    
    # Generate sample data
    elements = generate_sample_data()
    
    # Time the loop-based implementation
    import time
    
    start_time = time.time()
    j_loops = compute_j_integral_loops(elements)
    loop_time = time.time() - start_time
    
    print(f"Loop-based J-integral:     {j_loops:.6f}")
    print(f"Loop-based time:           {loop_time:.6f} seconds")
    
    # Count valid integration points
    valid_count = 0
    total_count = 0
    for elem in elements:
        if elem.on_integration_path:
            for gp in elem.gauss_points:
                total_count += 1
                if gp.state == 'intact':
                    valid_count += 1
    
    print(f"Number of valid integration points: {valid_count}/{total_count}")
    
    print("\nLoop-based implementation completed successfully!")

if __name__ == "__main__":
    main() 