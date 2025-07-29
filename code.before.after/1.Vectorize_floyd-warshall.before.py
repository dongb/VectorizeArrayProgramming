import numpy as np
import time

def floyd_warshall_naive(graph):
    n = graph.shape[0]
    dist = graph.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    return dist

def create_test_graph(n=5, seed=42):
    """Create a test graph with random weights for Floyd-Warshall algorithm."""
    np.random.seed(seed)
    
    # Create adjacency matrix with random weights
    graph = np.random.randint(1, 20, size=(n, n)).astype(float)
    
    # Set diagonal to 0 (distance from node to itself)
    np.fill_diagonal(graph, 0)
    
    # Set some edges to infinity to represent no direct connection
    mask = np.random.random((n, n)) > 0.7
    graph[mask] = np.inf
    np.fill_diagonal(graph, 0)  # Keep diagonal as 0
    
    return graph

def main():
    """Main function to test the naive Floyd-Warshall implementation."""
    print("Floyd-Warshall Algorithm: Naive Loop-based Implementation")
    print("=" * 60)
    
    # Test with different graph sizes
    test_sizes = [5, 10, 20]
    
    for n in test_sizes:
        print(f"\nTesting with graph size: {n}x{n}")
        print("-" * 30)
        
        # Create test graph
        graph = create_test_graph(n)
        
        print("Original graph:")
        print(graph)
        print()
        
        # Test naive implementation
        start_time = time.time()
        result_naive = floyd_warshall_naive(graph)
        time_naive = time.time() - start_time
        
        print(f"Naive implementation time: {time_naive:.6f} seconds")
        
        print("\nShortest distances matrix (naive result):")
        print(result_naive)
        
        if n <= 10:  # Only show detailed results for smaller matrices
            print("\nDetailed result:")
            print("Naive result:")
            print(result_naive)
    
    print("\n" + "=" * 60)
    print("Loop-based implementation complete!")

if __name__ == "__main__":
    main() 