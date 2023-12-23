from copy import deepcopy
import matplotlib
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

# Load the datasets
for n_bags in [30, 40, 50]:
    if n_bags == 30:
        n_bags_name = "extra-large"
    elif n_bags == 40:
        n_bags_name = "extra-extra-large"
    elif n_bags == 50:
        n_bags_name = "massive"

    #base_dataset = "cifar-10"
    base_dataset = "svhn"
    hard_dataset = f"{base_dataset}-hard-{n_bags_name}-fol-clust-fol-clust-cluster-kmeans-autoencoder-{n_bags}"
    intermediate_dataset = f"{base_dataset}-intermediate-{n_bags_name}-fol-clust-fol-clust-cluster-kmeans-autoencoder-{n_bags}"
    simple_dataset = f"{base_dataset}-simple-{n_bags_name}-fol-clust-fol-clust-cluster-kmeans-autoencoder-{n_bags}"
    naive_dataset = f"{base_dataset}-naive-{n_bags_name}-fol-clust-None-cluster-kmeans-autoencoder-{n_bags}"

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
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rc('font', size=6)

    fig, ax = plt.subplots(2, 2, figsize=(7.5, 5))
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

    ax[0, 0].set_ylabel("Bags")
    ax[0, 1].set_ylabel("Bags")
    ax[1, 0].set_xlabel("Bags")
    ax[1, 1].set_xlabel("Bags")

    plt.suptitle(f"{base_dataset} with {n_bags} bags")

    plt.tight_layout()
    plt.savefig(f"{base_dataset}-proportions-{n_bags}.png", dpi=800)
    plt.close()


        






