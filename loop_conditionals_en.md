
# Conditional Branching Can Be Vectorized: Efficient Logic with NumPy Broadcasting

## Abstract

Conditional logic is prevalent in data processing, image filtering, rule-based engines, and model preprocessing. While Python users often use explicit for-loops and if-else statements to implement such logic, these methods suffer from poor performance and readability on large datasets. This article demonstrates how to fully vectorize such operations using NumPy’s broadcasting and logical expressions, highlighting two practical problems and comparing traditional loop-based implementations with their vectorized counterparts.

---

## 1. Introduction

Although NumPy is designed for efficient numerical computation, many developers revert to loops when implementing logic with conditional branches. Particularly with multi-branch logic, it may seem that NumPy cannot express such structure easily.

This post introduces a “loop-to-broadcast” comparison framework and aims to help readers internalize how to express conditional logic in fully vectorized NumPy style.

---

## 2. Example 1: ReLU + Threshold Clipping

### 2.1 Problem Definition

Given a matrix $X \in \mathbb{R}^{m \times n}$, clip each element as follows:

```python
if x < 0:
    x = 0
elif x > T:
    x = T
else:
    x = x
```

### 2.2 For-loop Implementation

```python
def relu_clip_loop(X, T):
    m, n = X.shape
    result = np.empty_like(X)
    for i in range(m):
        for j in range(n):
            x = X[i, j]
            if x < 0:
                result[i, j] = 0
            elif x > T:
                result[i, j] = T
            else:
                result[i, j] = x
    return result
```

### 2.3 Vectorized Implementation

```python
def relu_clip_vectorized(X, T):
    return np.minimum(np.maximum(X, 0), T)
```

---

## 3. Example 2: Mask Selection with Broadcasted Weighting

### 3.1 Problem Definition

Given:

- Matrix $A \in \mathbb{R}^{m \times n}$
- Row vector $r \in \mathbb{R}^m$
- Column vector $c \in \mathbb{R}^n$
- Threshold $\theta \in \mathbb{R}$

Define:

```math
val_{ij} = A_{ij} + r_i \cdot c_j
```
```math
mask_{ij} =
\begin{cases}
1 & \text{if } val_{ij} > \theta \\
0 & \text{otherwise}
\end{cases}
```

### 3.2 For-loop Implementation

```python
def conditional_mask_loop(A, r, c, threshold):
    m, n = A.shape
    result = np.zeros_like(A, dtype=int)
    for i in range(m):
        for j in range(n):
            val = A[i, j] + r[i] * c[j]
            if val > threshold:
                result[i, j] = 1
    return result
```

### 3.3 Vectorized Implementation

```python
def conditional_mask_vectorized(A, r, c, threshold):
    val = A + r[:, np.newaxis] * c[np.newaxis, :]
    return (val > threshold).astype(int)
```

---

## 4. General Patterns for Vectorizing Conditional Logic

| Conditional Structure      | Vectorized Form                                      |
|----------------------------|------------------------------------------------------|
| Single condition replacement | `np.where(condition, A, B)`                       |
| Double threshold clip       | `np.minimum(np.maximum(X, lower), upper)`         |
| Multiple conditions         | `np.select([cond1, cond2, ...], [val1, val2, ...])` |
| Boolean mask update         | `X[mask] = value`                                  |
| Broadcasted condition matrix| `A + r[:, None] * c[None, :] > threshold`         |

---

## 5. Conclusion

We’ve seen how conditional logic structures commonly implemented with nested for-loops can be transformed into clean, high-performance NumPy vectorized expressions using broadcasting and logical functions. Whenever possible, developers should adopt vectorized design first rather than optimize loop code after the fact.

---

## Future Directions

- Port logic to GPU using PyTorch or TensorFlow
- Extend logic using `np.select` for multi-branch rules
- Apply in data preprocessing, image masks, scoring/ranking systems
