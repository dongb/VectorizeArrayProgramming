
# Vectorized Computation of Euclidean Distance Between Point Sets: From Loops to NumPy Broadcasting

## Introduction

In many scientific computing and machine learning tasks, computing **Euclidean distance matrices between point sets** is a fundamental yet frequently encountered operation. It is common in clustering analysis, nearest neighbor search (KNN), graph construction, and more.

Beginners often use two nested for-loops to compute this matrix, but such an approach becomes inefficient as the data size grows. This article walks through the derivation and implementation of a **vectorized and loop-free solution using NumPy broadcasting**, starting from the mathematical definition.

---

## Problem Definition

Given two sets of points:

- $A \in \mathbb{R}^{m \times d}$: $m$ points in $d$-dimensional space
- $B \in \mathbb{R}^{n \times d}$: $n$ points in $d$-dimensional space

We want to compute the distance matrix $D \in \mathbb{R}^{m \times n}$, where each entry is:

```math
D[i, j] = \|A_i - B_j\|_2
```

That is, the Euclidean distance between the \(i\)-th point in \(A\) and the \(j\)-th point in \(B\).

---

## Method 1: Using Nested For-Loops

```python
import numpy as np

def pairwise_distances_loop(A, B):
    m, n = A.shape[0], B.shape[0]
    D = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = np.linalg.norm(A[i] - B[j])
    return D
```

Example input:

```python
A = np.array([[0, 0], [1, 1], [2, 2]])
B = np.array([[1, 0], [2, 0]])
```

Output:

```
[[1.         2.23606798]
 [0.         1.41421356]
 [1.41421356 2.        ]]
```

---

## Method 2: Vectorized with Broadcasting

### Mathematical Formulation

We can express the Euclidean norm as:

```math
\|A_i - B_j\|_2 = \sqrt{ \sum_{k=1}^d (A_{ik} - B_{jk})^2 }
```

This formulation lends itself to tensor operations, which can be efficiently computed using broadcasting in NumPy.

### Implementation

```python
def pairwise_distances_broadcast(A, B):
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # shape: (m, n, d)
    D = np.linalg.norm(diff, axis=2)  # shape: (m, n)
    return D
```

Key broadcasting behavior:

- `A[:, np.newaxis, :]` → shape `(m, 1, d)`
- `B[np.newaxis, :, :]` → shape `(1, n, d)`
- Result of subtraction → shape `(m, n, d)`
- Norm along last axis gives distance matrix of shape `(m, n)`

---

## Dimensional Illustration

If \(m = 3\), \(n = 2\), and \(d = 2\):

```python
A.shape = (3, 2)
B.shape = (2, 2)
```

Broadcasting results in:

```python
A[:, np.newaxis, :]  # → (3, 1, 2)
B[np.newaxis, :, :]  # → (1, 2, 2)
Subtraction result    # → (3, 2, 2)
```

Each `diff[i, j, :]` is \(A_i - B_j\); `np.linalg.norm(..., axis=2)` gives the full distance matrix.

---

## Verification

```python
A = np.array([[0, 0], [1, 1], [2, 2]])
B = np.array([[1, 0], [2, 0]])

res_loop = pairwise_distances_loop(A, B)
res_bcast = pairwise_distances_broadcast(A, B)

print(np.allclose(res_loop, res_bcast))  # True
```

---

## Summary

This article demonstrates how to replace a double for-loop with efficient broadcasting using NumPy. The key lies in understanding input tensor shapes and how to manipulate them for bulk operations. This method is broadly applicable in:

- Computing squared distances without square roots
- High-dimensional distance calculations
- Generalizing to other metrics (e.g., Manhattan, cosine)

---

Further extensions include GPU acceleration (e.g., PyTorch, TensorFlow) and applying these techniques to kernel methods and graph learning tasks.
