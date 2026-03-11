import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Dataset
X = np.array([[1,2],[1,4],[1,0],
              [10,2],[10,4],[10,0]])

# Model
kmeans = KMeans(n_clusters=2)

# Train model
kmeans.fit(X)

# Prediction
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Clusters:", labels)
print("Centroids:", centroids)

# Plot
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centroids[:,0], centroids[:,1], marker='x')
plt.show()