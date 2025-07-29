# Vectorize Array Programming - Before & After Examples

This directory contains **before and after** implementations of various array programming problems, demonstrating the transformation from loop-based code to vectorized NumPy implementations.

## 📁 Directory Structure

Each programming problem has been split into two files:
- **`.before.py`** - Loop-based implementation (traditional approach)
- **`.after.py`** - Vectorized implementation (NumPy approach)

```
code.before.after/
├── 0.Vectorize_loop_with_conditionals.before.py  # Loop-based conditional operations
├── 0.Vectorize_loop_with_conditionals.after.py   # Vectorized conditional operations
├── 1.Vectorize_euclidean_distance.before.py      # Loop-based distance calculations
├── 1.Vectorize_euclidean_distance.after.py       # Broadcasting-based distances
├── 1.Vectorize_floyd-warshall.before.py          # Triple-loop Floyd-Warshall
├── 1.Vectorize_floyd-warshall.after.py           # Vectorized Floyd-Warshall
├── 2.Vectorize_six_loops.before.py               # Six nested loops
├── 2.Vectorize_six_loops.after.py                # Vectorized tensor operations
├── 2.Vectorize_triple_loops.before.py            # Matrix chain multiplication loops
├── 2.Vectorize_triple_loops.after.py             # Vectorized matrix chain
├── 3.Vectorize_j_integral.before.py              # Loop-based J-integral
├── 3.Vectorize_j_integral.after.py               # Vectorized J-integral
├── 3.Vectorize_spring_energy.before.py           # Loop-based energy calculation
├── 3.Vectorize_spring_energy.after.py            # Vectorized energy calculation
├── validate_equivalence.py                       # Validation script
└── README.md                                      # This file
```

## 🎯 Learning Objectives

By comparing the `.before.py` and `.after.py` versions, you'll learn:

1. **Performance Optimization** - How vectorization dramatically improves speed
2. **Code Simplification** - How NumPy reduces code complexity
3. **Memory Efficiency** - How broadcasting avoids unnecessary memory allocation
4. **Pythonic Programming** - How to write more elegant and maintainable code

## 📋 Examples Overview

### 0. Loop with Conditionals
**Problem**: Apply conditional operations (ReLU, clipping) and create conditional masks
- **Before**: Nested loops with if-else statements
- **After**: `np.minimum`, `np.maximum`, boolean indexing

### 1. Euclidean Distance Calculation
**Problem**: Compute pairwise distances between two sets of points
- **Before**: Double nested loops with `np.linalg.norm`
- **After**: Broadcasting with `A[:, np.newaxis, :] - B[np.newaxis, :, :]`

### 1. Floyd-Warshall Algorithm
**Problem**: Find shortest paths in a weighted graph
- **Before**: Triple nested loops for all-pairs shortest paths
- **After**: Vectorized operations with `np.minimum` and broadcasting

### 2. Six Nested Loops
**Problem**: Compute a 6D tensor with pairwise distance sums
- **Before**: Six nested loops creating O(N^6) tensor
- **After**: Precompute pairwise distances, then use broadcasting

### 2. Triple Loop Matrix Chain
**Problem**: Dynamic programming for optimal matrix multiplication order
- **Before**: Triple nested loops for cost computation
- **After**: Vectorized inner loop using `np.arange` and array operations

### 3. J-Integral Computation
**Problem**: Fracture mechanics path integral calculation
- **Before**: Element and Gauss point loops with conditionals
- **After**: Vectorized operations with `np.einsum` and boolean masking

### 3. Spring Energy Calculation
**Problem**: Compute pairwise spring energy between particles
- **Before**: Double nested loops for energy matrix
- **After**: Broadcasting to compute all pairs simultaneously

## 🚀 Running the Examples

Each file can be run independently:

```bash
# Run loop-based version
python 0.Vectorize_loop_with_conditionals.before.py

# Run vectorized version
python 0.Vectorize_loop_with_conditionals.after.py

# Compare performance and results
python 1.Vectorize_euclidean_distance.before.py
python 1.Vectorize_euclidean_distance.after.py
```

## ✅ Verification of Equivalence

To ensure all `.before.py` and `.after.py` implementations produce identical results, run the validation script:

```bash
# Validate all before/after pairs
python validate_equivalence.py
```

### What the Validation Script Does

The `validate_equivalence.py` script automatically:

- **Loads all before/after implementation pairs**
- **Runs them with identical test data** (same random seeds)
- **Compares results with high precision** (tolerance: 1e-10 to 1e-12)
- **Reports detailed test results** for each implementation pair
- **Provides a comprehensive summary** with pass/fail statistics

### Expected Output

When all implementations are correct, you should see:

```
🧪 Validating Equivalence of Before/After Implementations
============================================================
🔍 Validating: Loop with Conditionals...
   ReLU Clip: ✅ PASS
   Conditional Mask: ✅ PASS

🔍 Validating: Euclidean Distance...
   Pairwise Distances: ✅ PASS

[... more validations ...]

============================================================
📊 VALIDATION SUMMARY
============================================================
Total Tests: 7
Passed: 7
Failed: 0
Success Rate: 100.0%

🎉 ALL TESTS PASSED! All before/after implementations are equivalent.
```

### Why This Matters

This validation ensures:
- **Numerical Correctness**: Vectorized versions produce identical results
- **Learning Confidence**: You can trust the performance comparisons
- **Code Quality**: No bugs were introduced during vectorization
- **Educational Value**: Focus on learning techniques, not debugging errors

## 📊 Performance Expectations

Expected speedup factors (approximate):
- **Conditional Operations**: 10-50x faster
- **Distance Calculations**: 20-100x faster  
- **Floyd-Warshall**: 5-20x faster
- **Six Loops**: 100-1000x faster
- **Matrix Chain**: 2-10x faster
- **J-Integral**: 50-200x faster
- **Spring Energy**: 50-500x faster

*Actual speedups depend on problem size and hardware*

## 🧠 Key Vectorization Techniques Demonstrated

### 1. Broadcasting
```python
# Before: explicit loops
for i in range(m):
    for j in range(n):
        result[i, j] = A[i] + B[j]

# After: broadcasting
result = A[:, np.newaxis] + B[np.newaxis, :]
```

### 2. Boolean Indexing
```python
# Before: conditional loops
for i in range(len(arr)):
    if arr[i] > threshold:
        result[i] = 1

# After: boolean indexing
result = (arr > threshold).astype(int)
```

### 3. Einsum Operations
```python
# Before: manual tensor contractions
for i in range(N):
    for j in range(M):
        result[i] += A[i, k] * B[k, j]

# After: einsum
result = np.einsum('ik,kj->ij', A, B)
```

### 4. Axis-wise Operations
```python
# Before: manual reductions
for i in range(shape[0]):
    result[i] = np.sum(matrix[i, :])

# After: axis operations
result = np.sum(matrix, axis=1)
```

## 💡 Study Recommendations

1. **Verify First**: Run `python validate_equivalence.py` to ensure all implementations work correctly
2. **Start Small**: Begin with `0.Vectorize_loop_with_conditionals`
3. **Compare Side-by-Side**: Open both `.before.py` and `.after.py` files
4. **Run Timing Tests**: Notice the dramatic performance differences
5. **Understand Broadcasting**: This is the key to most vectorizations
6. **Practice**: Try vectorizing your own loop-based code

## 🔍 Common Patterns

### Pattern 1: Element-wise Operations
```python
# Before: explicit loops for element operations
# After: direct array operations

# Before: for i in range(len(x)): y[i] = f(x[i])
# After: y = f(x)
```

### Pattern 2: Pairwise Operations
```python
# Before: double loops for all pairs
# After: broadcasting with np.newaxis

# Before: for i: for j: result[i,j] = f(a[i], b[j])
# After: result = f(a[:, np.newaxis], b[np.newaxis, :])
```

### Pattern 3: Conditional Processing
```python
# Before: if-else in loops
# After: boolean indexing and np.where

# Before: for i: if condition[i]: result[i] = value
# After: result[condition] = value
```

## ⚠️ Important Notes

- **Memory Usage**: Vectorized operations may use more memory temporarily
- **Problem Size**: Benefits are most apparent with larger arrays
- **Readability**: Sometimes loops are clearer than complex vectorized code
- **Debugging**: Vectorized code can be harder to debug step-by-step
- **Verification**: Run `validate_equivalence.py` if you have any doubts about correctness

## 🎓 Next Steps

After mastering these examples:
1. Learn advanced NumPy features like `np.einsum` and `np.tensordot`
2. Explore specialized libraries like SciPy and scikit-learn
3. Consider GPU acceleration with CuPy or JAX for even greater speedups

---

**Happy Vectorizing! 🚀**

*Remember: The goal isn't always to eliminate every loop, but to use the right tool for each task.* 