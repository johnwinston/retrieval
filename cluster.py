import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Example embeddings
with open("./data/embeddings.json", "r") as f:
    embeddings = np.array(json.load(f))

with open("./data/descriptions.json", "r") as f:
    descriptions = json.load(f)

# Print the first description
print(list(descriptions.values())[0])

# Print the 20th description
print(list(descriptions.values())[20])

print(list(descriptions.values())[105])

'''
with open("./data/ontology_embeddings.json", "r") as f:
    ontology_embeddings = json.load(f)

def collect_embeddings(nested_dict, embeddings_list):
    for key, value in nested_dict.items():
        # Check if the current value is a dictionary
        if isinstance(value, dict):
            # Recursive call to explore deeper
            collect_embeddings(value, embeddings_list)
        elif key == "embedding":
            # Add the embedding to the list
            embeddings_list.append(value)

embeddings = []
collect_embeddings(description_embeddings, embeddings)
embeddings = np.array(embeddings)
'''

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=0)
reduced_embeddings = tsne.fit_transform(embeddings)

#reduced_embeddings = embeddings

# Clustering
kmeans = KMeans(n_clusters=3, random_state=0)  # Choose an appropriate number of clusters
clusters = kmeans.fit_predict(reduced_embeddings)

# Visualization
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters)
# Add numbers to the points
for i in range(len(reduced_embeddings)):
    plt.annotate(i, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
plt.colorbar()
plt.show()
