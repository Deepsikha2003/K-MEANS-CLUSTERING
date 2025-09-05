import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def visualize_clusters(df, kmeans_model, scaler):
    """
    Creates a scatter plot to visualize the customer clusters.
    
    Args:
        df (pd.DataFrame): The original DataFrame with cluster labels.
        kmeans_model (KMeans): The trained K-means model.
        scaler (StandardScaler): The scaler used for data preprocessing.
    """
    # Get the unscaled cluster centers
    centroids = scaler.inverse_transform(kmeans_model.cluster_centers_)
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', 
                    hue='Cluster', data=df, palette='viridis', s=100)
    
    # Plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', 
                marker='X', label='Centroids')
    
    plt.title('Customer Segments')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid(True)
    plt.show()