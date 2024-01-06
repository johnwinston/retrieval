import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Example embeddings
with open("./data/embeddings.json", "r") as f:
    embeddings = np.array(json.load(f))

print(len(embeddings))

# Applying t-SNE
#tsne = TSNE(n_components=2, random_state=0)
#reduced_embeddings = tsne.fit_transform(embeddings)
reduced_embeddings = embeddings

# Clustering
kmeans = KMeans(n_clusters=3, random_state=0)  # Choose an appropriate number of clusters
clusters = kmeans.fit_predict(reduced_embeddings)

# Visualization
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters)
plt.colorbar()
plt.show()
