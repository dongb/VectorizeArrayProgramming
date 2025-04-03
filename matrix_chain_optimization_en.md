# Efficient Implementation of Matrix Chain Multiplication: From Triple Loops to Vectorization and Parenthesis Recovery

## Abstract

Matrix Chain Multiplication is a classical dynamic programming problem that aims to determine the optimal parenthesization of a matrix product sequence such that the total number of scalar multiplications is minimized. This article begins with the mathematical formulation and naive triple-loop implementation, then introduces a NumPy-based vectorized approach, and finally extends the implementation with parenthesis reconstruction support.

---

## 1. Problem Description and Mathematical Model

Given a sequence of matrices $A_1, A_2, \dots, A_n$, where matrix $A_i$ has dimensions $p_{i-1} 	imes p_i$, the goal is to determine the optimal order of multiplication using the associativity of matrix multiplication.

Define $m[i][j]$ as the minimum number of scalar multiplications required to compute the product $A_i \cdot A_{i+1} \cdots A_j$. The recurrence relation is:

```math
m[i][j] = \min_{i \leq k < j} \left\{ m[i][k] + m[k+1][j] + p_i \cdot p_{k+1} \cdot p_{j+1} \right\}
```
With base condition:

```math
m[i][i] = 0, \quad orall i
```

The final result is \( m[0][n-1] \), the minimum cost to multiply the entire matrix chain.

---

## 2. Naive Triple Loop Implementation

```python
import numpy as np

def matrix_chain_order_naive(p):
    n = len(p) - 1
    m = np.zeros((n, n))
    
    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            m[i][j] = np.inf
            for k in range(i, j):
                cost = m[i][k] + m[k+1][j] + p[i] * p[k+1] * p[j+1]
                if cost < m[i][j]:
                    m[i][j] = cost
    return m
```

---

## 3. Vectorized Optimization

We can vectorize the inner `k` loop by precomputing all possible values and using `np.min` to select the best cost:

```python
def matrix_chain_order_vectorized(p):
    n = len(p) - 1
    m = np.full((n, n), np.inf)
    np.fill_diagonal(m, 0)

    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            k_vals = np.arange(i, j)
            left = m[i, k_vals]
            right = m[k_vals + 1, j]
            mult_cost = p[i] * p[k_vals + 1] * p[j + 1]
            total_cost = left + right + mult_cost
            m[i, j] = np.min(total_cost)
    
    return m
```

---

## 4. Further Optimization with Path Recovery

To reconstruct the parenthesis structure, we store the optimal `k` that yields the minimum cost:

```python
def matrix_chain_order_optimized(p):
    n = len(p) - 1
    m = np.full((n, n), np.inf)
    s = np.full((n, n), -1, dtype=int)
    np.fill_diagonal(m, 0)

    for l in range(2, n + 1):
        for i in range(n - l + 1):
            j = i + l - 1
            k_vals = np.arange(i, j)
            left = m[i, k_vals]
            right = m[k_vals + 1, j]
            mult_cost = p[i] * p[k_vals + 1] * p[j + 1]
            total_cost = left + right + mult_cost
            best_k_idx = np.argmin(total_cost)
            m[i, j] = total_cost[best_k_idx]
            s[i, j] = k_vals[best_k_idx]
    
    return m, s
```

### Function to build parenthesis structure:

```python
def build_optimal_parens(s, i, j):
    if i == j:
        return f"A{i+1}"
    else:
        k = s[i][j]
        left = build_optimal_parens(s, i, k)
        right = build_optimal_parens(s, k + 1, j)
        return f"({left} x {right})"
```

### Example:

```python
p = [30, 35, 15, 5, 10, 20, 25]
m, s = matrix_chain_order_optimized(p)

print("Minimum multiplication cost:", m[0, len(p)-2])
print("Optimal parenthesization:", build_optimal_parens(s, 0, len(p)-2))
```

---

## 5. Structural and Performance Comparison

| Feature                  | Triple Loop               | Vectorized (k loop)         |
|--------------------------|---------------------------|------------------------------|
| Implementation Complexity | Simple and direct         | Slightly more complex        |
| Inner Loop Performance   | Slow in Python             | Faster with NumPy vector ops |
| Extendibility            | Easy to add path tracking  | Achievable via separate matrix |
| Parallelization Potential| Limited                    | High (e.g., JAX, Numba)      |

---

## Conclusion

We demonstrated a complete optimization path for Matrix Chain Multiplication: from mathematical modeling to efficient vectorized implementation with path reconstruction. By identifying the structure of the recurrence relation, we directly constructed vector operations, avoiding inefficient Python-level loops. This method is widely applicable to other DP problems with similar repetitive structure.
