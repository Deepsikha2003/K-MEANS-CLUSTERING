import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    """
    Loads, selects features, and scales the mall customer data.
    
    Args:
        file_path (str): The path to the CSV dataset.
        
    Returns:
        tuple: A tuple containing the original DataFrame and the scaled features.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Select the features for clustering
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X_scaled