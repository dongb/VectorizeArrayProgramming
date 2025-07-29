#!/usr/bin/env python3
"""
Validation Script for Before/After Implementations

This script validates that all .before.py and .after.py implementations
produce equivalent results by running them with identical test data.
"""

import numpy as np
import sys
import importlib.util
import os
from typing import Tuple, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_module_from_file(filepath: str, module_name: str):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def validate_conditionals() -> bool:
    """Validate 0.Vectorize_loop_with_conditionals"""
    print("🔍 Validating: Loop with Conditionals...")
    
    try:
        # Load modules
        before_mod = load_module_from_file("0.Vectorize_loop_with_conditionals.before.py", "conditionals_before")
        after_mod = load_module_from_file("0.Vectorize_loop_with_conditionals.after.py", "conditionals_after")
        
        # Test data
        np.random.seed(42)
        X = np.random.randn(5, 4) * 10
        T = 5.0
        A = np.random.randn(4, 3) * 2
        r = np.random.randn(4)
        c = np.random.randn(3)
        threshold = 1.0
        
        # Test ReLU clip
        result_before_relu = before_mod.relu_clip_loop(X, T)
        result_after_relu = after_mod.relu_clip_vectorized(X, T)
        
        # Test conditional mask
        result_before_mask = before_mod.conditional_mask_loop(A, r, c, threshold)
        result_after_mask = after_mod.conditional_mask_vectorized(A, r, c, threshold)
        
        # Validate
        relu_match = np.allclose(result_before_relu, result_after_relu, rtol=1e-10)
        mask_match = np.array_equal(result_before_mask, result_after_mask)
        
        success = relu_match and mask_match
        print(f"   ReLU Clip: {'✅ PASS' if relu_match else '❌ FAIL'}")
        print(f"   Conditional Mask: {'✅ PASS' if mask_match else '❌ FAIL'}")
        
        return success
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def validate_euclidean_distance() -> bool:
    """Validate 1.Vectorize_euclidean_distance"""
    print("🔍 Validating: Euclidean Distance...")
    
    try:
        before_mod = load_module_from_file("1.Vectorize_euclidean_distance.before.py", "distance_before")
        after_mod = load_module_from_file("1.Vectorize_euclidean_distance.after.py", "distance_after")
        
        np.random.seed(42)
        A = np.random.rand(5, 3)
        B = np.random.rand(4, 3)
        
        result_before = before_mod.pairwise_distances_loop(A, B)
        result_after = after_mod.pairwise_distances_broadcast(A, B)
        
        success = np.allclose(result_before, result_after, rtol=1e-10)
        print(f"   Pairwise Distances: {'✅ PASS' if success else '❌ FAIL'}")
        
        if not success:
            print(f"   Max difference: {np.max(np.abs(result_before - result_after))}")
        
        return success
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def validate_floyd_warshall() -> bool:
    """Validate 1.Vectorize_floyd-warshall"""
    print("🔍 Validating: Floyd-Warshall...")
    
    try:
        before_mod = load_module_from_file("1.Vectorize_floyd-warshall.before.py", "floyd_before")
        after_mod = load_module_from_file("1.Vectorize_floyd-warshall.after.py", "floyd_after")
        
        # Create test graph
        graph = before_mod.create_test_graph(5, seed=42)
        
        result_before = before_mod.floyd_warshall_naive(graph)
        result_after = after_mod.floyd_warshall_numpy(graph)
        
        success = np.allclose(result_before, result_after, rtol=1e-10)
        print(f"   Floyd-Warshall Algorithm: {'✅ PASS' if success else '❌ FAIL'}")
        
        if not success:
            print(f"   Max difference: {np.max(np.abs(result_before - result_after))}")
        
        return success
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def validate_six_loops() -> bool:
    """Validate 2.Vectorize_six_loops"""
    print("🔍 Validating: Six Loops...")
    
    try:
        before_mod = load_module_from_file("2.Vectorize_six_loops.before.py", "six_before")
        after_mod = load_module_from_file("2.Vectorize_six_loops.after.py", "six_after")
        
        np.random.seed(42)
        N = 3  # Small size for faster computation
        D = 2
        X = np.random.randn(N, D)
        
        result_before = before_mod.tensor_six_naive(X)
        result_after = after_mod.tensor_six_vectorized(X)
        
        success = np.allclose(result_before, result_after, rtol=1e-10, atol=1e-10)
        print(f"   Six-dimensional Tensor: {'✅ PASS' if success else '❌ FAIL'}")
        
        if not success:
            print(f"   Max difference: {np.max(np.abs(result_before - result_after))}")
        
        return success
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def validate_triple_loops() -> bool:
    """Validate 2.Vectorize_triple_loops"""
    print("🔍 Validating: Triple Loops (Matrix Chain)...")
    
    try:
        before_mod = load_module_from_file("2.Vectorize_triple_loops.before.py", "triple_before")
        after_mod = load_module_from_file("2.Vectorize_triple_loops.after.py", "triple_after")
        
        test_cases = [
            [1, 2, 3, 4, 5],
            [5, 4, 6, 2, 7],
            [40, 20, 30, 10, 30]
        ]
        
        all_success = True
        for i, p in enumerate(test_cases):
            result_before = before_mod.matrix_chain_order_naive(p)
            result_after = after_mod.matrix_chain_order_vectorized(p)
            
            # Only compare upper triangular part (what matters for matrix chain)
            upper_before = np.triu(result_before)
            upper_after = np.triu(result_after)
            
            success = np.allclose(upper_before, upper_after, rtol=1e-10)
            print(f"   Test Case {i+1}: {'✅ PASS' if success else '❌ FAIL'}")
            
            if not success:
                print(f"   Max difference: {np.max(np.abs(upper_before - upper_after))}")
                all_success = False
        
        return all_success
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def validate_j_integral() -> bool:
    """Validate 3.Vectorize_j_integral"""
    print("🔍 Validating: J-Integral...")
    
    try:
        before_mod = load_module_from_file("3.Vectorize_j_integral.before.py", "j_before")
        after_mod = load_module_from_file("3.Vectorize_j_integral.after.py", "j_after")
        
        # Generate same data for both
        elements = before_mod.generate_sample_data(num_elements=10, gauss_per_elem=4)
        W, sigma, grad_u, n, weights, valid_mask = after_mod.generate_sample_data(num_elements=10, gauss_per_elem=4)
        
        result_before = before_mod.compute_j_integral_loops(elements)
        result_after = after_mod.compute_j_integral_vectorized(W, sigma, grad_u, n, weights, valid_mask)
        
        success = np.allclose(result_before, result_after, rtol=1e-10, atol=1e-12)
        print(f"   J-Integral Computation: {'✅ PASS' if success else '❌ FAIL'}")
        
        if not success:
            print(f"   Before: {result_before}")
            print(f"   After: {result_after}")
            print(f"   Difference: {abs(result_before - result_after)}")
        
        return success
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def validate_spring_energy() -> bool:
    """Validate 3.Vectorize_spring_energy"""
    print("🔍 Validating: Spring Energy...")
    
    try:
        before_mod = load_module_from_file("3.Vectorize_spring_energy.before.py", "spring_before")
        after_mod = load_module_from_file("3.Vectorize_spring_energy.after.py", "spring_after")
        
        np.random.seed(42)
        X = np.random.rand(6, 2) * 10
        
        result_before = before_mod.pairwise_energy_naive(X)
        result_after = after_mod.pairwise_energy_vectorized(X)
        
        success = np.allclose(result_before, result_after, rtol=1e-10)
        print(f"   Pairwise Spring Energy: {'✅ PASS' if success else '❌ FAIL'}")
        
        if not success:
            print(f"   Max difference: {np.max(np.abs(result_before - result_after))}")
        
        return success
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 Validating Equivalence of Before/After Implementations")
    print("=" * 60)
    
    # Store original directory and change to script directory
    original_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        validation_functions = [
            validate_conditionals,
            validate_euclidean_distance,
            validate_floyd_warshall,
            validate_six_loops,
            validate_triple_loops,
            validate_j_integral,
            validate_spring_energy
        ]
        
        results = []
        for validate_func in validation_functions:
            try:
                result = validate_func()
                results.append(result)
                print()
            except Exception as e:
                print(f"❌ CRITICAL ERROR in {validate_func.__name__}: {e}")
                results.append(False)
                print()
        
        # Summary
        print("=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        passed = sum(results)
        total = len(results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        
        if all(results):
            print("\n🎉 ALL TESTS PASSED! All before/after implementations are equivalent.")
            return True
        else:
            print("\n⚠️  SOME TESTS FAILED! Check the output above for details.")
            return False
            
    finally:
        # Restore original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 