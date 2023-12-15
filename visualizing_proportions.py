from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def compute_proportions(bags, y):
    """Compute the proportions for each bag given.
    Parameters
    ----------
    bags : {array-like}
    y : {array-like}

    Returns
    -------
    proportions : {array}
        An array of type np.float
    """
    n_classes = len(np.unique(y))
    num_bags = len(np.unique(bags))

    if n_classes == 2:
        proportions = np.empty(num_bags, dtype=float)
        for i in range(num_bags):
            bag = np.where(bags == i)[0]
            proportions[i] = np.count_nonzero(y[bag] == 1) / len(bag)
    else:
        proportions = np.empty((num_bags, n_classes), dtype=float)
        for i in range(num_bags):
            bag = np.where(bags == i)[0]
            for j in range(n_classes):
                proportions[i, j] = np.count_nonzero(y[bag] == j) / len(bag)

    return proportions

# Set the style
sns.set_style('whitegrid')

# Load the datasets
base_dataset = "cifar-10"
hard_dataset = "cifar-10-hard-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40"
intermediate_dataset = "cifar-10-intermediate-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40"
simple_dataset = "cifar-10-simple-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40"
naive_dataset = "cifar-10-naive-extra-extra-large-fol-clust-None-cluster-kmeans-autoencoder-40"

# Reading X, y (base dataset) and bags (dataset)
df = pd.read_parquet("datasets-ci/" + base_dataset + ".parquet")
y = deepcopy(df["y"].values)
y = y.reshape(-1)

proportions_hard, proportions_intermediate, proportions_simple, proportions_naive = None, None, None, None
for dataset in [hard_dataset, intermediate_dataset, simple_dataset, naive_dataset]:
    df = pd.read_parquet(f"datasets-ci/{dataset}.parquet")
    bags = df["bag"].values
    bags = bags.reshape(-1)
    proportions = compute_proportions(bags, y)
    if dataset == hard_dataset:
        proportions_hard = proportions
    elif dataset == intermediate_dataset:
        proportions_intermediate = proportions
    elif dataset == simple_dataset:
        proportions_simple = proportions
    elif dataset == naive_dataset:
        proportions_naive = proportions


# Plot the heatmap of the cosine similarity matrix of the proportions (subplots for each dataset)
fig, ax = plt.subplots(2, 2, figsize=(15, 5))
# Compute the cosine similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity_matrix_hard = cosine_similarity(proportions_hard)
cosine_similarity_matrix_intermediate = cosine_similarity(proportions_intermediate)
cosine_similarity_matrix_simple = cosine_similarity(proportions_simple)
cosine_similarity_matrix_naive = cosine_similarity(proportions_naive)

# Plot the heatmap
sns.heatmap(cosine_similarity_matrix_hard, ax=ax[0, 0], cmap="Blues", vmin=0.0, vmax=1.0)
sns.heatmap(cosine_similarity_matrix_intermediate, ax=ax[0, 1], cmap="Blues", vmin=0.0, vmax=1.0)
sns.heatmap(cosine_similarity_matrix_simple, ax=ax[1, 0], cmap="Blues", vmin=0.0, vmax=1.0)
sns.heatmap(cosine_similarity_matrix_naive, ax=ax[1, 1], cmap="Blues", vmin=0.0, vmax=1.0)

# Set the titles
ax[0, 0].set_title("Hard")
ax[0, 1].set_title("Intermediate")
ax[1, 0].set_title("Simple")
ax[1, 1].set_title("Naive")

# Set the labels
ax[0, 0].set_xlabel("Bags")
ax[0, 0].set_ylabel("Bags")
ax[0, 1].set_xlabel("Bags")
ax[0, 1].set_ylabel("Bags")
ax[1, 0].set_xlabel("Bags")
ax[1, 0].set_ylabel("Bags")
ax[1, 1].set_xlabel("Bags")
ax[1, 1].set_ylabel("Bags")

plt.show()

        






