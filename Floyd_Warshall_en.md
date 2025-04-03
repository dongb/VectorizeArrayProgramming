
# From Recurrence Relation to Efficient Implementation: A Stepwise Vectorization of the Floyd-Warshall Algorithm

## Introduction

In graph theory, the All-Pairs Shortest Path (APSP) problem is fundamental and widely applied. Its goal is to compute the shortest path distances between all pairs of vertices in a weighted graph. The Floyd-Warshall algorithm is a classical solution based on dynamic programming, with a well-defined recurrence structure and good implementability.

While the traditional Floyd-Warshall algorithm is implemented using triple nested loops, modern numerical environments allow us to optimize performance using vectorized computation frameworks such as NumPy. This article derives an efficient vectorized implementation directly from the mathematical formulation, avoiding the common workflow of writing loop-based code first and optimizing later.

---

## Problem Formulation and Mathematical Recurrence

Given a directed graph `G = (V, E)`, where:

- Vertex set: `V = {1, 2, ..., n}`
- Weight function:

$$
w: V \times V \rightarrow \mathbb{R} \cup \{+\infty\}
$$

`w(i, j)` denotes the weight of the edge from node `i` to node `j`. We aim to construct a shortest path function `d(i, j)` representing the minimum distance from `i` to `j`.

The Floyd-Warshall recurrence is defined as:

$$
d_k(i, j) = \min\left( d_{k-1}(i, j),\ d_{k-1}(i, k) + d_{k-1}(k, j) \right)
$$

where `d_k(i, j)` denotes the shortest path using only intermediate nodes from `{1, ..., k}`.

Initial condition:

$$
d_0(i, j) = 
\begin{cases}
0 & \text{if } i = j \\
w(i, j) & \text{if } i \ne j
\end{cases}
$$

The final result `d_n(i, j)` gives the minimum distance between all pairs of nodes.

---

## Baseline Implementation: Triple Loop Structure

A direct implementation of the recurrence using three nested loops is:

```python
import numpy as np

def floyd_warshall_naive(graph):
    n = graph.shape[0]
    dist = graph.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    return dist
```

Although logically clear and faithful to the formula, this implementation is inefficient due to high-level Python loop overhead, especially when `n` is large.

---

## Vectorized Implementation: Broadcasting from Recurrence

Observing the core update:

$$
d(i, j) = \min\left( d(i, j),\ d(i, k) + d(k, j) \right)
$$

In each iteration, `k` is fixed, and the update for all pairs `(i, j)` can be done in parallel. This allows us to reformulate the recurrence as a matrix operation.

For fixed `k`, define:

- Column vector `c = d[:, k] ∈ ℝⁿ`
- Row vector `r = d[k, :] ∈ ℝⁿ`

Then all `d(i, k) + d(k, j)` values form an outer sum matrix:

```python
d[:, k, np.newaxis] + d[k, :]
```

We update as:

```python
d = np.minimum(d, d[:, k, np.newaxis] + d[k, :])
```

---

## NumPy Implementation

Full vectorized implementation using broadcasting:

```python
def floyd_warshall_numpy(graph):
    n = graph.shape[0]
    dist = graph.copy()
    for k in range(n):
        dist = np.minimum(dist, dist[:, k, np.newaxis] + dist[k, :])
    return dist
```

Only a single loop over `k` is retained, and the rest is handled efficiently using NumPy broadcasting.

---

## Test and Verification

```python
inf = np.inf
graph = np.array([
    [0,   3,   inf, 7],
    [8,   0,   2,   inf],
    [5,   inf, 0,   1],
    [2,   inf, inf, 0]
], dtype=float)

# Compare both implementations
res1 = floyd_warshall_naive(graph)
res2 = floyd_warshall_numpy(graph)

assert np.allclose(res1, res2)
```

The outputs from both methods should be identical.

---

## Comparison

| Aspect           | Triple Loop Version        | NumPy Vectorized Version    |
|------------------|----------------------------|-----------------------------|
| Implementation   | Three nested for loops     | Single loop + broadcasting  |
| Performance      | `O(n³)`, slow               | `O(n³)`, faster constant    |
| Readability      | Intuitive, math-aligned     | Requires broadcasting knowledge |
| Extensibility    | Easy to add path recovery  | Needs additional tracking   |

---

## Conclusion: From Math to Efficient Code, Directly

Rather than writing inefficient code and optimizing later, we recommend the following development path:

1. **Start from the mathematical recurrence**
2. **Analyze the dimensional structure**
3. **Construct a matrix-based formulation**
4. **Apply broadcasting directly in implementation**

This results in faster development, better performance, and cleaner code.

---

Future extensions may include path reconstruction, negative cycle detection, or sparse graph optimizations.
