# Replacing Six Nested Loops with NumPy Broadcasting: A High-Dimensional Tensor Example

## Introduction

In high-dimensional data analysis, graph modeling, kernel construction, and molecular simulations, it is often necessary to compute relative distances or interactions between multiple combinations of points. These problems frequently involve deeply nested loops to cover all combinations. While conceptually straightforward, such implementations are highly inefficient in Python.

This article presents a canonical example: constructing a 6D tensor using pairwise distances, and shows how to transform the six-loop implementation into a clean, efficient, fully vectorized NumPy solution using broadcasting.

---

## Problem Description

Given a set of points $X = [x_0, x_1, ..., x_{N-1}] \in \mathbb{R}^{N \times D}$, where each point $x_i$ is a D-dimensional vector, we want to construct a six-dimensional tensor:

```math
T[i, j, k, l, m, n] = \|x_i - x_j\|^2 + \|x_k - x_l\|^2 + \|x_m - x_n\|^2
```

Each tensor element represents the sum of squared distances over three point-pairs. This structure encodes higher-order relationships, useful in six-body interactions, tensor kernels, or relational graph modeling.

---

## Naive Implementation: Six Nested Loops

```python
import numpy as np

def tensor_six_naive(X):
    N = X.shape[0]
    T = np.zeros((N, N, N, N, N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    for m in range(N):
                        for n in range(N):
                            T[i,j,k,l,m,n] = (
                                np.sum((X[i] - X[j])**2) +
                                np.sum((X[k] - X[l])**2) +
                                np.sum((X[m] - X[n])**2)
                            )
    return T
```

This implementation is computationally expensive with time complexity $O(N^6 \cdot D)$, and becomes infeasible for $N > 6$.

---

## Vectorized Solution with Broadcasting

We can observe that the tensor expression is separable:

```math
T[i, j, k, l, m, n] = A[i, j] + A[k, l] + A[m, n]
```

Where $A$ is the matrix of pairwise squared distances between all points.

### Step 1: Compute Pairwise Squared Distances

```python
def pairwise_sq_dists(X):
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # shape (N, N, D)
    return np.sum(diff ** 2, axis=2)                  # shape (N, N)
```

### Step 2: Expand and Broadcast Tensors

```python
def tensor_six_vectorized(X):
    A = pairwise_sq_dists(X)  # shape (N, N)

    term1 = A[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    term2 = A[np.newaxis, np.newaxis, :, :, np.newaxis, np.newaxis]
    term3 = A[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]

    T = term1 + term2 + term3  # shape (N, N, N, N, N, N)
    return T
```

---

## Example Execution

```python
np.random.seed(0)
X = np.random.rand(4, 3)

T_naive = tensor_six_naive(X)
T_vec = tensor_six_vectorized(X)

print(np.allclose(T_naive, T_vec))  # Output: True
```

---

## Comparison

| Feature            | Naive (Loops)        | Vectorized (Broadcast)     |
|--------------------|----------------------|-----------------------------|
| Time Complexity     | $O(N^6 \cdot D)$ | $O(N^2 \cdot D + N^6)$   |
| Space Complexity    | $O(N^6)$          | Same                        |
| Python Overhead     | High                | Low                         |
| Readability         | High (for teaching) | Medium (requires NumPy)     |

> ⚠️ Note: Full tensor construction with shape \(N^6\) is memory-intensive. Even \(N=10\) results in ~1GB memory usage. This method is practical for small sample sizes and high-order modeling needs.

---

## Summary

This article demonstrates how to convert a six-level nested loop for tensor construction into a three-part broadcasting expression using NumPy. This pattern applies broadly to higher-order tensor operations, particularly when the structure is decomposable.

This transformation is particularly effective when:
- Each term is independently computable.
- The tensor shape can be split across multiple broadcastable axes.
- The final result is a combination (sum/product) of independent components.

If you'd like to extend this to sparse variants, masked sampling, or batch-wise computation, feel free to reach out or adapt from this base.

