# Vectorization Techniques in Computational Science

This repository contains comprehensive examples demonstrating the transformation from nested loops to vectorized implementations using NumPy. It provides both educational materials and hands-on code examples to master array programming techniques.

## 🏗️ Repository Structure

```
VectorizeArrayProgramming/
├── code.before.after/          # 🎯 Split implementations for easy comparison
│   ├── *.before.py             # Loop-based implementations
│   ├── *.after.py              # Vectorized implementations  
│   ├── validate_equivalence.py # Verification script
│   └── README.md               # Detailed learning guide
├── code/                       # 📚 Original combined examples
├── *.md                        # 📖 Theoretical explanations
└── README.md                   # This file
```

## 🎯 Two Learning Approaches

### 1. **Before/After Comparison** (Recommended for beginners)
The `code.before.after/` directory provides clear side-by-side comparisons:
- **`.before.py`** - Traditional loop-based approach
- **`.after.py`** - Optimized vectorized version
- **Perfect for learning** - See exact transformations step by step

### 2. **Combined Examples** 
The `code/` directory contains comprehensive examples with both approaches in single files.

## 📋 Covered Examples

- **Conditional Operations**: ReLU, clipping, and conditional masking
- **Euclidean Distance Calculation**: Pairwise distances between point sets
- **Floyd-Warshall Algorithm**: All-pairs shortest path optimization  
- **Matrix Chain Multiplication**: Dynamic programming vectorization
- **High-Dimensional Tensors**: Six-loop optimization using broadcasting
- **J-Integral Calculation**: Fracture mechanics computational optimization
- **Spring Energy Systems**: Pairwise interaction energy computation


## 📖 Theoretical Documentation

The following markdown files provide in-depth theoretical explanations and background:

### 1. `Vectorize_floyd-warshall.md`
- **Topic**: Efficient implementation of the Floyd-Warshall algorithm.
- **Summary**: The article compares the classical triple-loop implementation with the vectorized solution using NumPy broadcasting, optimizing performance for all-pairs shortest path computation.

### 2. `Vectorize_euclidean_distance.md`
- **Topic**: Vectorized computation of Euclidean distances.
- **Summary**: This article shows how to efficiently compute a distance matrix between two sets of points using NumPy's broadcasting, replacing the traditional double `for` loop approach.

### 3. `Vectorize_j_integral.md`
- **Topic**: Vectorization of the J-integral for fracture mechanics.
- **Summary**: The J-integral is calculated using a path-independent contour integral. This article demonstrates how to vectorize this computation using NumPy for efficiency in finite element analysis.

### 4. `Vectorize_spring_energy.md`
- **Topic**: Pairwise spring energy computation.
- **Summary**: A mechanical system with pairwise spring interactions is optimized using NumPy to compute an energy matrix efficiently, replacing a nested loop approach with broadcasting.

### 5. `Vectorize_triple_loops.md`
- **Topic**: Optimizing matrix chain multiplication.
- **Summary**: This file showcases the transformation of a triple nested loop dynamic programming problem into a vectorized solution with NumPy.

### 6. `Vectorize_six_loops.md`
- **Topic**: High-dimensional tensor construction using pairwise distances.
- **Summary**: The example demonstrates how to compute a six-dimensional tensor of pairwise distances efficiently with broadcasting, replacing six nested loops.

### 7. `Vectorize_loop_with_conditionals.md`
- **Topic**: Vectorization of conditional logic in loops.
- **Summary**: This file explains how to replace nested loops with conditional logic in a vectorized form using NumPy, including examples like ReLU clipping and mask selection.

### 8. `Understanding_np_newaxis.md`
- **Topic**: Understanding the `np.newaxis` operation.
- **Summary**: This article explores the geometric intuition behind `np.newaxis` and its role in defining tensor structure for broadcasting in NumPy.

### 9. `Understand_numpy_broadcast.md`
- **Topic**: Broadcasting in NumPy.
- **Summary**: A deep dive into NumPy's broadcasting capabilities, including examples and applications like matrix chain multiplication and the Floyd-Warshall algorithm.

## 🚀 Quick Start

### For Beginners (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dongb/VectorizeArrayProgramming.git
   cd VectorizeArrayProgramming
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy
   ```

3. **Start with before/after comparisons:**
   ```bash
   cd code.before.after
   
   # Verify all implementations work correctly
   python validate_equivalence.py
   
   # Try a simple example
   python 0.Vectorize_loop_with_conditionals.before.py
   python 0.Vectorize_loop_with_conditionals.after.py
   ```

4. **Read the learning guide:**
   ```bash
   # Open the detailed README in code.before.after/
   cat code.before.after/README.md
   ```

### For Advanced Users

Explore the comprehensive examples in the `code/` directory and theoretical explanations in the markdown files.

## ✅ Verification

This repository includes a comprehensive validation system:

```bash
# Verify all before/after implementations produce identical results
cd code.before.after
python validate_equivalence.py
```

**All 7 implementation pairs are verified to be mathematically equivalent**.

## 📊 Expected Performance Improvements

Typical speedup factors from vectorization:
- **Conditional Operations**: 10-50x faster
- **Distance Calculations**: 20-100x faster  
- **Floyd-Warshall**: 5-20x faster
- **Six-loop Tensors**: 100-1000x faster
- **Matrix Chain**: 2-10x faster
- **J-Integral**: 50-200x faster
- **Spring Energy**: 50-500x faster

*Results vary with problem size and hardware*

## 🧠 Key Concepts

- **Vectorization**: Converting operations into array-level computations to eliminate explicit loops
- **Broadcasting**: NumPy's ability to perform operations on arrays of different shapes without explicit data replication  
- **Element-wise Operations**: Applying functions to entire arrays simultaneously
- **Boolean Indexing**: Using boolean arrays to filter and select data efficiently
- **Memory Layout Optimization**: Leveraging contiguous memory access patterns for better performance

## 🎓 Learning Path

### Beginner → Intermediate → Advanced

1. **Start Here**: `code.before.after/0.Vectorize_loop_with_conditionals`
2. **Build Understanding**: Work through all before/after examples
3. **Dive Deeper**: Read theoretical documentation (*.md files)
4. **Practice**: Apply techniques to your own problems
5. **Verify**: Use validation script to check your implementations

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Add new vectorization examples
- Improve documentation
- Report issues or suggest improvements
- Share your own before/after transformations

## 📄 License

This project is open source and available under the [Apache License 2.0](LICENSE).

---

**Happy Vectorizing! 🚀** 

*"The best way to learn vectorization is to see the transformation happen step by step."*
