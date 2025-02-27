from sklearn.cluster import KMeans
import numpy as np
# Data points
data = np.array([[1, 2], [1, 4], [10, 2], [10, 4]])
# Apply K-Means with random_state and n_init set explicitly
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(data)
# Output labels
print("Labels:", kmeans.labels_)
