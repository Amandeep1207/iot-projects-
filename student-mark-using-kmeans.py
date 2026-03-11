import numpy as np
from sklearn.cluster import KMeans

# Dataset (Student Marks)
marks = np.array([[40],[45],[50],[55],[60],[65],[70],[75],[80],[85]])

# KMeans Model
kmeans = KMeans(n_clusters=3)

# Train model
kmeans.fit(marks)

# Cluster prediction
clusters = kmeans.labels_

print("Student Marks:", marks.flatten())
print("Clusters:", clusters)

# Cluster centers
print("Cluster Centers:", kmeans.cluster_centers_)