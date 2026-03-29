from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# Create blobs
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)

points = data[0]
labels = data[1]

feature_1 = points[:, 0]
feature_2 = points[:, 1]

# importing the library
from sklearn.cluster import KMeans

# K values to test (explicit list instead of range)
k_values = np.array([2, 3, 4, 5, 6])

for k in k_values:
    plt.subplot(1, 2, 1)
    plt.scatter(feature_1, feature_2, c=labels, s=10, cmap='viridis')
    plt.title('True Clusters')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    # plt.show()

    # Create the KMeans object and specify its characteristics
    kmeans_method = KMeans(n_clusters=int(k))
    kmeans_method.fit(points)

    # Predict the cluster for a set of points
    labels_predicted = kmeans_method.predict(points)

    plt.subplot(1, 2, 2)
    plt.scatter(feature_1, feature_2, c=labels_predicted, s=10, cmap='viridis')
    plt.title('Predicted Clusters')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.show()
