import argparse

from datasets import llp_variant_generation

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from datasets import llp_variant_generation

from autoencoder.cifar_grey_autoencoder import CIFARGreyAE
from autoencoder.cifar_autoencoder import CIFARAE
from autoencoder.svhn_autoencoder import SVHNAE

import torch

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
parser = argparse.ArgumentParser()
parser.add_argument("--base_dataset", "-bd", type=str, required=True)
parser.add_argument("--clustering_method", "-cm", type=str, required=True)
parser.add_argument("--n_bags", "-nb", type=str, required=True, choices=["extra-large", "extra-extra-large", "massive"])
args = parser.parse_args()

base_dataset = args.base_dataset
clustering_method = args.clustering_method
n_bags_type = args.n_bags

if base_dataset == "adult":
    if n_bags_type == "extra-large":
        n_bags = n_clusters = 20
    elif n_bags_type == "extra-extra-large":
        n_bags = n_clusters = 25
    elif n_bags_type == "massive":
        n_bags = n_clusters = 30
    else:
        raise Exception("ERROR: The number of bags is not correct")
elif base_dataset == "cifar-10-grey-animal-vehicle":
    if n_bags_type == "extra-large":
        n_bags = n_clusters = 30
    elif n_bags_type == "extra-extra-large":
        n_bags = n_clusters = 40
    elif n_bags_type == "massive":
        n_bags = n_clusters = 50
    else:
        raise Exception("ERROR: The number of bags is not correct")
elif base_dataset == "cifar-10":
    if n_bags_type == "extra-large":
        n_bags = n_clusters = 30
    elif n_bags_type == "extra-extra-large":
        n_bags = n_clusters = 40
    elif n_bags_type == "massive":
        n_bags = n_clusters = 50
    else:
        raise Exception("ERROR: The number of bags is not correct")
elif base_dataset == "svhn":
    if n_bags_type == "extra-large":
        n_bags = n_clusters = 30
    elif n_bags_type == "extra-extra-large":
        n_bags = n_clusters = 40
    elif n_bags_type == "massive":
        n_bags = n_clusters = 50
    else:
        raise Exception("ERROR: The number of bags is not correct")
    
bags_size_type = proportions_type = "fol-clust"

X, y = load_base_dataset(base_dataset, random_state=random_state)
clusters, latent_representation = get_clusters(X, clustering_method, n_clusters, base_dataset, random_state)

proportions_target, bags_sizes_target = compute_proportions(clusters, y), np.bincount(clusters)
bags_sizes_target_naive = np.bincount(clusters)

n_classes = len(np.unique(y))

if n_classes == 2:
    if (np.isclose(proportions_target, 0)).any() or (np.isclose(proportions_target, 1)).any():
        raise Exception("ERROR: The proportions given by the cluster are not correct.")
else:
    if np.isclose(proportions_target, np.zeros(proportions_target.shape)).all(axis=1).any() or np.isclose(proportions_target, np.ones(proportions_target.shape)).all(axis=1).any():
        raise Exception("ERROR: The proportions given by the cluster are not correct.")

# Saving the base datasets (it will be the same for all variants)
df_base = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
df_base["y"] = y
df_base["y"] = df_base["y"].astype(int)
df_base.to_parquet("datasets-ci/{}.parquet".format(base_dataset), index=False)

#### Intermediate #####
llp_variant = "intermediate"

# Bags as clusters
bags = clusters
proportions_found = deepcopy(proportions_target)
bags_size_found = bags_sizes_target

# Changing a bit the target to not make bags with proportion 0 (in this case, in the Hard variant)
if base_dataset == "adult" and n_bags_type == "massive":
    print("Changing proportions_target for adult-massive (Simple and Hard)")
    proportions_target[14] += 0.01 

if base_dataset == "cifar-10-grey-animal-vehicle" and n_bags_type == "extra-extra-large":
    print("Changing proportions_target for cifar-10-grey-animal-vehicle-extra-extra-large (Simple and Hard)")
    proportions_target = np.round(proportions_target, 2)

# Saving only the bags (saving space)
df = pd.DataFrame(bags, columns=["bag"], dtype=int)

filename = "datasets-ci/{}-{}-{}-{}-{}-cluster-{}-{}.parquet".format(base_dataset, llp_variant, n_bags_type, \
                    bags_size_type, 
                    proportions_type, \
                    clustering_method, \
                    n_clusters)
df.to_parquet(filename, index=False)
print("Dataset {} generated".format(filename))
print("Proportions: {}".format(proportions_found))
print("Bag sizes: {}".format(bags_size_found))
print("\n------------------------\n")


##########################################
all_proportions = {"intermediate": proportions_found}

# Generating the datasets 
for llp_variant in VARIANTS:
    if llp_variant == "intermediate":
        continue
    
    bags = generate_llp_dataset(X, y, clusters, proportions_target, bags_sizes_target if llp_variant != "naive" else bags_sizes_target_naive, llp_variant, random_state)
    unique_bags, bags_sizes = np.unique(bags, return_counts=True)
    if len(unique_bags) != n_bags:
        print("\n\n")
        print(base_dataset)
        print(llp_variant)
        print(n_bags_type)
        print(bags_size_type)
        print(proportions_type)
        print(unique_bags)
        print(n_bags)
        raise Exception("ERROR: The number of bags is not correct")

    all_proportions[llp_variant] = compute_proportions(bags, y)
    
    if n_classes == 2:
        if (np.isclose(all_proportions[llp_variant], 0)).any() or (np.isclose(all_proportions[llp_variant], 1)).any():            
            print("\n\n")
            print(base_dataset)
            print(llp_variant)
            print(n_bags_type)
            print(bags_size_type)
            print(proportions_type)
            print(all_proportions[llp_variant])
            raise Exception("ERROR: The proportions are not correct")
    else:
        if np.isclose(all_proportions[llp_variant], np.zeros(all_proportions[llp_variant].shape)).all(axis=1).any() or np.isclose(all_proportions[llp_variant], np.ones(all_proportions[llp_variant].shape)).all(axis=1).any():
            print("\n\n")
            print(base_dataset)
            print(llp_variant)
            print(n_bags_type)
            print(bags_size_type)
            print(proportions_type)
            print(all_proportions[llp_variant])
            raise Exception("ERROR: The proportions are not correct")

    # Saving only the bags (saving space)
    df = pd.DataFrame(bags, columns=["bag"], dtype=int)
    
    filename = "datasets-ci/{}-{}-{}-{}-{}-cluster-{}-{}.parquet".format(base_dataset, llp_variant, n_bags_type, \
                        bags_size_type, 
                        proportions_type if llp_variant != "naive" else "None", \
                        clustering_method, \
                        n_clusters)
    df.to_parquet(filename, index=False)
    print("Dataset {} generated".format(filename))
    print("Proportions: {}".format(compute_proportions(bags, y)))
    print("Bag sizes: {}".format(np.bincount(bags)))
    print("\n------------------------\n")

for i in range(len(VARIANTS)):
    for j in range(i+1, len(VARIANTS)):
        v1 = VARIANTS[i]
        v2 = VARIANTS[j]
        if v1 != v2:
            print("Proportions {} vs {}: {}".format(v1, v2, np.max(np.abs(all_proportions[v1] - all_proportions[v2]))))