import argparse

from datasets import llp_variant_generation

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from datasets import llp_variant_generation

from autoencoder.cifar_grey_autoencoder import CIFARGreyAE
from autoencoder.cifar_autoencoder import CIFARAE
from autoencoder.svhn_autoencoder import SVHNAE

import torch

import time

def get_latent_representation(X, base_dataset):
    print("Computing latent representation...")
    trainset = torch.from_numpy(X).float()
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
    if base_dataset == "cifar-10-grey-animal-vehicle":
        model = CIFARGreyAE()
    elif base_dataset == "cifar-10":
        model = CIFARAE()
    elif base_dataset == "svhn":
        model = SVHNAE()
    else:
        raise Exception("ERROR: Base dataset has no autoencoder")

    # generate latent vectors and save to file
    if base_dataset == "cifar-10-grey-animal-vehicle":
        model.load_state_dict(torch.load('autoencoder/weights-cifar-10-grey-animal-vehicle.pth'))
    elif base_dataset == "cifar-10":
        model.load_state_dict(torch.load('autoencoder/weights-cifar-10.pth'))
    elif base_dataset == "svhn":
        model.load_state_dict(torch.load('autoencoder/weights-svhn.pth'))
    model.eval()

    if base_dataset == "cifar-10-grey-animal-vehicle":
        with torch.no_grad():
            latent_data = torch.concat([model(data.reshape(1, 32, 32))[0].flatten().reshape(1, -1) for data in train_loader]).numpy()
    elif base_dataset == "cifar-10" or base_dataset == "svhn":
        with torch.no_grad():
            latent_data = torch.concat([model(data.reshape(3, 32, 32))[0].flatten().reshape(1, -1) for data in train_loader]).numpy()

    return latent_data


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

def load_base_dataset(base_dataset, random_state=None):
    """
    Load the base dataset and return the X and y.

    Parameters
    ----------
    base_dataset : {str}
        The base dataset to load.
    random_state : {int}
        The random state to use.

    Returns
    -------
    X : {array-like}
        The features.
    y : {array-like}
        The labels.
    """
    if base_dataset == "adult":
        df = pd.read_parquet("base-datasets/adult.parquet")
        df.drop(columns=["educational-num"], inplace=True)
        df.gender.replace({"Male": 0, "Female": 1}, inplace=True)
        df.income.replace({"<=50K": 0, ">50K": 1}, inplace=True)
        df = pd.get_dummies(df, columns=
                            ["age", "workclass", "education", "marital-status", \
                             "occupation", "relationship", "race", "native-country"])
        df = df.sample(frac=1, random_state=random_state)
        X = deepcopy(df.drop(columns=["income"]).values)
        y = deepcopy(df["income"].values)
    elif base_dataset == "cifar-10":
        df = pd.read_parquet("base-datasets/cifar-10.parquet")
        df = df.sample(frac=1, random_state=random_state)
        X = deepcopy(df.drop(columns=["label"]).values)
        y = deepcopy(df["label"].values)
    elif base_dataset == "cifar-10-grey-animal-vehicle":
        df = pd.read_parquet("base-datasets/cifar-10-grey.parquet")
        df = df.sample(frac=1, random_state=random_state)
        df.label = df.label.apply(lambda x: 0 if x in [0, 1, 8, 9] else 1)
        X = deepcopy(df.drop(columns=["label"]).values)
        y = deepcopy(df["label"].values)
    elif base_dataset == "svhn":
        df = pd.read_parquet("base-datasets/svhn.parquet")
        df = df.sample(frac=1, random_state=random_state)
        X = deepcopy(df.drop(columns=["label"]).values)
        y = deepcopy(df["label"].values)
    
    X = MinMaxScaler().fit_transform(X)

    return X, y

def get_clusters(X, clustering_method, n_clusters, base_dataset, random_state):
    if clustering_method == "kmeans":
        latent_representation = []
        clusters = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state).fit_predict(X)
    elif clustering_method == "kmeans-autoencoder":
        latent_representation = get_latent_representation(X, base_dataset)
        clusters = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state).fit_predict(latent_representation)

    # If some clusters are empty, we terminate the program
    for i in range(n_clusters):
        if np.count_nonzero(clusters == i) == 0:
            print("Some clusters are empty. Please try again with a different clustering method.")
            exit()

    return clusters, latent_representation

def generate_llp_dataset(X, y, clusters, proportions_target, bags_sizes_target, llp_variant, random_state):
    if llp_variant == "naive":
        bags = llp_variant_generation(X, llp_variant="naive", bags_size_target=bags_sizes_target, 
                                    random_state=random_state)
    elif llp_variant == "simple":
        bags = llp_variant_generation(X, y, llp_variant="simple", bags_size_target=bags_sizes_target, 
                                    proportions_target=proportions_target, 
                                    random_state=random_state)
    elif llp_variant == "intermediate":
        bags = llp_variant_generation(X, y, llp_variant="intermediate", bags_size_target=bags_sizes_target, 
                                    proportions_target=proportions_target, clusters=clusters, 
                                    random_state=random_state)
    elif llp_variant == "hard":
        bags = llp_variant_generation(X, y, llp_variant="hard", bags_size_target=bags_sizes_target, 
                                    proportions_target=proportions_target, clusters=clusters, 
                                    random_state=random_state)
    return bags

random_state=6738921
random = np.random.RandomState(random_state)
n_jobs=-1
VARIANTS = ["naive", "simple", "intermediate", "hard"]

# Argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--base_dataset", "-bd", type=str, required=True)
# parser.add_argument("--clustering_method", "-cm", type=str, required=True)
# parser.add_argument("--n_bags", "-nb", type=str, required=True, choices=["extra-large", "extra-extra-large", "massive"])
# args = parser.parse_args()

# base_dataset = args.base_dataset
# clustering_method = args.clustering_method
# n_bags_type = args.n_bags
base_dataset = "make_classification"
clustering_method = "kmeans"

# Parameters
n_bags = 1000
n_clusters = 50
bags_size_type = proportions_type = "fol-clust"

#X, y = load_base_dataset(base_dataset, random_state=random_state)

from sklearn.datasets import fetch_kddcup99

X, y = fetch_kddcup99(return_X_y=True, percent10=True)

# One hot encoding
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X = enc.fit_transform(X).toarray()
print(X.shape)

start = time.time()
clusters, latent_representation = get_clusters(X, clustering_method, n_clusters, base_dataset, random_state)
end = time.time()
print("Time to cluster: ", end - start)

proportions_target, bags_sizes_target = compute_proportions(clusters, y), np.bincount(clusters)
bags_sizes_target_naive = np.bincount(clusters)

n_classes = len(np.unique(y))

if n_classes == 2:
    if (np.isclose(proportions_target, 0)).any() or (np.isclose(proportions_target, 1)).any():
        raise Exception("ERROR: The proportions given by the cluster are not correct.")
else:
    if np.isclose(proportions_target, np.zeros(proportions_target.shape)).all(axis=1).any() or np.isclose(proportions_target, np.ones(proportions_target.shape)).all(axis=1).any():
        raise Exception("ERROR: The proportions given by the cluster are not correct.")

df_time = pd.DataFrame(columns=["base_dataset", "variant", "n_bags", "time"])

##########################################
# Generating the datasets 
for llp_variant in VARIANTS:
    time_start = time.time()
    bags = generate_llp_dataset(X, y, clusters, proportions_target, bags_sizes_target if llp_variant != "naive" else bags_sizes_target_naive, llp_variant, random_state)
    time_end = time.time()

    df_time = df_time.append({"base_dataset": base_dataset, "variant": llp_variant, "n_bags": n_bags, "time": time_end - time_start}, ignore_index=True)

print(df_time)
    

