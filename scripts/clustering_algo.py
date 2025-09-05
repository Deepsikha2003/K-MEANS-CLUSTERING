from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_optimal_k(data, max_k=10):
    """
    Uses the Elbow Method to find the optimal number of clusters.
    
    Args:
        data (np.ndarray): The scaled feature data.
        max_k (int): The maximum number of clusters to test.
        
    Returns:
        tuple: A tuple containing the optimal k and the elbow plot.
    """
    inertia = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    
    # Plot the Elbow Method graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.title('The Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()
    
    # Note: You'll need to manually inspect the plot to determine the optimal k.
    print("Please examine the plot to find the optimal k and update the 'run_clustering' function.")
    
    return inertia

def run_clustering(data, optimal_k):
    """
    Builds and fits the K-means model.
    
    Args:
        data (np.ndarray): The scaled feature data.
        optimal_k (int): The chosen number of clusters.
        
    Returns:
        KMeans: The trained KMeans model.
    """
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(data)
    return kmeans