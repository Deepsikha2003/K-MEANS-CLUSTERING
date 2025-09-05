K-Means Customer Segmentation
Project Overview
This project uses the unsupervised machine learning algorithm K-means clustering to segment customers of a retail store. The goal is to identify distinct customer groups based on their purchasing behavior, specifically their Annual Income and Spending Score, to enable targeted marketing strategies.

The entire analysis, from data preprocessing to visualization, is performed within a Jupyter Notebook.

Dataset
The analysis is based on the Mall Customer Dataset, which contains customer IDs, demographic information (Age, Gender), and key financial metrics (Annual Income (k$) and Spending Score (1-100)). The dataset can be found on Kaggle.

Project Structure
K-Means-Clustering/
├── data/
│   ├── Mall_Customers.csv
├── scripts/
│   ├── data_preprocessing.py
│   ├── clustering_algo.py
│   └── visualization.py
├── notebooks/
│   └── clustering_model.ipynb
├── visualizations/
│   ├── elbow_method.png
│   └── clusters.png
├── .gitignore
└── requirements.txt
Methodology
The entire workflow is executed within the clustering_model.ipynb Jupyter Notebook, which imports and utilizes functions from the scripts in the scripts/ folder.

Data Preprocessing: The data_preprocessing.py script was used to load the dataset, select the Annual Income and Spending Score features, and standardize them using StandardScaler. This is a critical step for K-means, as it ensures all features contribute equally to the distance calculation.

Optimal k Determination: The clustering_algo.py script employed the Elbow Method to find the optimal number of clusters. The method works by plotting the sum of squared distances of samples to their closest cluster center (inertia) for a range of k values. The "elbow" of the curve, where the inertia begins to decrease more slowly, indicates the optimal k. In this analysis, the optimal number of clusters was determined to be 5.

K-Means Clustering: The clustering_algo.py script was then used to train the K-means model with the optimal k. The model assigned a cluster label to each customer.

Visualization and Interpretation: The visualization.py script generated a scatter plot of the clusters. Each cluster was assigned a unique color, and the cluster centroids were marked. By analyzing the characteristics of each cluster, five distinct customer segments were identified:

Cluster 0 (Careful): Low income, low spending.

Cluster 1 (Target): Mid income, mid spending.

Cluster 2 (High Potential): Low income, high spending.

Cluster 3 (High Value): High income, high spending.

Cluster 4 (Low Engagement): High income, low spending.

Key Outcomes
Actionable Insights: The project successfully segmented the customer base into distinct, meaningful groups.

Business Impact: The identified segments can be used to develop targeted marketing campaigns, optimize product offerings, and improve customer retention strategies, leading to increased profitability.

How to Run the Project
Clone the repository.

Install the required libraries: pip install -r requirements.txt.

Navigate to the notebooks/ directory.

Open the clustering_model.ipynb notebook and run all the cells.