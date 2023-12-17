"""
----------------------------------------
Dataset: cifar-10-grey-animal-vehicle-naive-small-not-equal-None-cluster-None-None
Model: mixbag
Loss function: abs
Params: {'lr': [0.1, 0.01, 0.001, 0.0001, 1e-05]}
n_splits: 5
validation_size: 0.5
splitter: split-bag-shuffle
Execution: 1
----------------------------------------

Breaking bags (training) due to memory issues!!!
Before
Bags sizes: [13963 13725  6831  6963  3518]
Proportions: [0.60209124 0.59985428 0.60518226 0.60261382 0.61284821]

After
Bags sizes: [ 6982 13725  6831  6963  3518  6981]
Proportions: [0.60209124 0.59985428 0.60518226 0.60261382 0.61284821 0.60209124]

________________________________________________________________________________


----------------------------------------
Dataset: cifar-10-grey-animal-vehicle-simple-small-not-equal-close-global-cluster-None-None
Model: mixbag
Loss function: abs
Params: {'lr': [0.1, 0.01, 0.001, 0.0001, 1e-05]}
n_splits: 5
validation_size: 0.5
splitter: split-bag-shuffle
Execution: 0
----------------------------------------

Breaking bags (training) due to memory issues!!!
Before
Bags sizes: [13934 13743  6956  6871  3496]
Proportions: [0.7033874  0.51538965 0.70385279 0.41362247 0.69965675]

After
Bags sizes: [ 6967 13743  6956  6871  3496  6967]
Proportions: [0.7033874  0.51538965 0.70385279 0.41362247 0.69965675 0.7033874 ]


________________________________________________________________________________

----------------------------------------
Dataset: cifar-10-grey-animal-vehicle-simple-small-not-equal-mixed-cluster-None-None
Model: mixbag
Loss function: abs
Params: {'lr': [0.1, 0.01, 0.001, 0.0001, 1e-05]}
n_splits: 5
validation_size: 0.5
splitter: split-bag-shuffle
Execution: 1
----------------------------------------

Breaking bags (training) due to memory issues!!!
Before
Bags sizes: [14969 12283  6896  7253  3599]
Proportions: [0.92243971 0.12293414 0.56902552 0.74769061 0.68380106]

After
Bags sizes: [ 7485 12283  6896  7253  3599  7484]
Proportions: [0.92243971 0.12293414 0.56902552 0.74769061 0.68380106 0.92243971]

"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import time
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from llp_learn.em import EM
from llp_learn.dllp import DLLP
from llp_learn.mixbag import MixBag
from llp_learn.llpvat import LLPVAT
from llp_learn.llpfc import LLPFC

from grid_search_experiments import gridSearchCVExperiments
from almostnolabel import MM, LMM, AMM

import torchvision.transforms as transforms
import torch

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


VARIANTS = ["naive", "simple", "intermediate", "hard"]

def load_dataset(args, execution):
    # Extracting variant
    llp_variant = [variant for variant in VARIANTS if variant in args.dataset][0]
     
    # Extracting base dataset
    base_dataset = args.dataset.split(llp_variant)[0]
    base_dataset = base_dataset[:-1]
    base_dataset += ".parquet"

    # Reading X, y (base dataset) and bags (dataset)
    df = pd.read_parquet("datasets-ci/" + base_dataset)
    X = df.drop(["y"], axis=1).values
    y = df["y"].values
    y = y.reshape(-1)
    
    # In NN based methods, we use 0 and 1 as labels
    if not args.model in NN_BASED_METHODS:
        y[y == 0] = -1

    df = pd.read_parquet("datasets-ci/" + args.dataset + ".parquet")
    bags = df["bag"].values
    bags = bags.reshape(-1)

    train_index, test_index = next(ShuffleSplit(n_splits=1, test_size=0.25, random_state=seed[execution]).split(X))

    return X, bags, y, train_index, test_index

if __name__ == "__main__":
    # Constants
    n_executions = 5 # Number of executions

    try:
        N_JOBS = eval(os.getenv('NSLOTS'))
    except:
        N_JOBS = -1
    print("Using {} cores".format(N_JOBS))

    seed = [189395, 962432364, 832061813, 316313123, 1090792484,
            1041300646,  242592193,  634253792,  391077503, 2644570296, 
            1925621443, 3585833024,  530107055, 3338766924, 3029300153,
        2924454568, 1443523392, 2612919611, 2781981831, 3394369024,
            641017724,  626917272, 1164021890, 3439309091, 1066061666,
            411932339, 1446558659, 1448895932,  952198910, 3882231031]
    
    NN_BASED_METHODS = [
        "dllp", "mixbag", "llp-vat", "llpfc"
    ]

    # Experiments that we have to break the bags due to memory issues
    BREAKING_BAGS = [
        {
            "dataset": "cifar-10-grey-animal-vehicle-naive-small-not-equal-None-cluster-None-None",
            "model": "mixbag",
            "loss": "abs",
            "n_splits": 5,
            "validation_size": 0.5,
            "splitter": "split-bag-shuffle",
            "execution": 1
        },
        {
            "dataset": "cifar-10-grey-animal-vehicle-simple-small-not-equal-close-global-cluster-None-None",
            "model": "mixbag",
            "loss": "abs",
            "n_splits": 5,
            "validation_size": 0.5,
            "splitter": "split-bag-shuffle",
            "execution": 0
        },
        {
            "dataset": "cifar-10-grey-animal-vehicle-simple-small-not-equal-mixed-cluster-None-None",
            "model": "mixbag",
            "loss": "abs",
            "n_splits": 5,
            "validation_size": 0.5,
            "splitter": "split-bag-shuffle",
            "execution": 1
        },
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device: %s" % device)

    directory = "llp-benchmark-results/"

    # Parsing arguments
    parser = argparse.ArgumentParser(description="LLP benchmark experiments")
    parser.add_argument("--dataset", "-d", required=True, help="the dataset that will be used in the experiments")
    parser.add_argument("--model", "-m", choices=["kdd-lr", "lmm", "amm", "mm", "dllp", "mixbag", "llp-vat", "llpfc"], required=True,
                        help="the model that will be used in the experiments")
    parser.add_argument("--loss", "-l", choices=["abs"],
                        help="the loss function that will be used in the experiment")
    parser.add_argument("--n_splits", "-n", type=int,
                        help="the number of splits that will be used in the experiment")
    parser.add_argument("--validation_size", "-v", type=float,
                        help="the validation size that will be used in the experiment")
    parser.add_argument("--splitter", "-s", choices=["full-bag-stratified-k-fold", "split-bag-bootstrap", "split-bag-shuffle", "split-bag-k-fold"],
                        help="the splitter that will be used in the experiment")
    parser.add_argument("--execution", "-e", choices=[-1] + [x for x in range(n_executions)], type=int, required=True,
                        help="the execution of the experiment")
    args = parser.parse_args()

    try:
        os.mkdir(directory)
    except:
        pass

    if args.execution is not None:
        args.execution = int(args.execution)

    if args.execution == -1:
        executions = range(n_executions)
    else:
        executions = [args.execution]

    for execution in executions:
        start = time.time()

        filename = directory + str(args.dataset) + "_" + str(args.model) + "_" + str(
            args.loss) + "_" + str(None) + "_" + str(args.splitter) + "_" + str(args.n_splits) + "_" + str(args.validation_size) + "_" + str(execution) + ".parquet"

        if args.model == "kdd-lr":
            params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}
        elif args.model == "lmm":
            params = {"lmd": [0, 1, 10, 100], "gamma": [0.01, 0.1, 1], "sigma": [0.25, 0.5, 1]}
        elif args.model == "amm":
            params = {"lmd": [0, 1, 10, 100], "gamma": [0.01, 0.1, 1], "sigma": [1]}
        elif args.model == "mm":
            params = {"lmd": [0, 1, 10, 100]}
        elif args.model in NN_BASED_METHODS:
            params = {"lr": [0.1, 0.01, 0.001, 0.0001, 0.00001]}
        else:
            params = {"C": [0.1, 1, 10], "C_p": [1, 10, 100]}

        print("----------------------------------------")
        print("Dataset: %s" % args.dataset)
        print("Model: %s" % args.model)
        print("Loss function: %s" % args.loss)
        print("Params: %s" % params)
        print("n_splits: %s" % args.n_splits)
        print("validation_size: %s" % args.validation_size)
        print("splitter: %s" % args.splitter)
        print("Execution: %s" % execution)
        print("----------------------------------------\n")

        X, bags, y, train_index, test_index = load_dataset(args, execution)

        model_type = "simple-mlp"

        if "grey" in args.dataset:
            n_channels = 1
        else:
            n_channels = 3

        if "cifar" in args.dataset and args.model in NN_BASED_METHODS:
            X = X.reshape(X.shape[0], n_channels, 32, 32)
            X = X.transpose(0, 2, 3, 1)

            # Normalizing using pytorch normalization
            if n_channels == 1:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
                ])

            X = [transform(x) for x in X]
            X = torch.stack(X)
            X = X.float()

            X_train, y_train, bags_train = X[train_index], y[train_index], bags[train_index]
            X_test, y_test, bags_test = X[test_index], y[test_index], bags[test_index]

            model_type = "resnet18"
        else:
            X_train, y_train, bags_train = X[train_index], y[train_index], bags[train_index]
            X_test, y_test, bags_test = X[test_index], y[test_index], bags[test_index]

            scaler = MinMaxScaler((-1, 1))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            if args.model in NN_BASED_METHODS:
                X_train, X_test = X_train.astype("float32"), X_test.astype("float32")

        proportions = compute_proportions(bags_train, y_train)

        # Breaking bags due to memory issues. We are breaking the bag with the highest number of instances.
        for breaking_bag in BREAKING_BAGS:
            if args.dataset == breaking_bag["dataset"] and args.model == breaking_bag["model"] and args.loss == breaking_bag["loss"] and args.n_splits == breaking_bag["n_splits"] and args.validation_size == breaking_bag["validation_size"] and args.splitter == breaking_bag["splitter"] and execution == breaking_bag["execution"]:
                print("Breaking bags (training) due to memory issues!!!")
                _, bags_counts = np.unique(bags_train, return_counts=True)
                bag_break = np.argmax(bags_counts)
                bag_break_idx = np.where(bags_train == bag_break)[0]
                # Using random seed
                random = np.random.RandomState(seed[execution])
                random.shuffle(bag_break_idx)
                bag_break_idx = bag_break_idx[:len(bag_break_idx) // 2]
                new_bag_idx = np.max(bags_train) + 1
                bags_train[bag_break_idx] = new_bag_idx
                # Not recomputing the proportions after breaking the bags
                print("Before")
                print(f"Bags sizes: {bags_counts}")
                print(f"Proportions: {proportions}")
                print()

                proportions = list(proportions)
                proportions.append(proportions[bag_break])
                proportions = np.array(proportions)


                print("After")
                print(f"Bags sizes: {np.unique(bags_train, return_counts=True)[1]}")
                print(f"Proportions: {proportions}")

        df_results = pd.DataFrame(columns=["metric", "accuracy_train", "accuracy_test", "f1_train", "f1_test", "best_hyperparams"])

        print("Model type: %s" % model_type)

        print("Execution started!!!")

        if args.model == "kdd-lr":
            model = EM(LogisticRegression(solver='lbfgs'), init_y="random", random_state=seed[execution])
        elif args.model == "lmm":
            model = LMM(lmd=1, gamma=1, sigma=1)
        elif args.model == "amm":
            model = AMM(lmd=1, gamma=1, sigma=1)
        elif args.model == "mm":
            model = MM(lmd=1)
        elif args.model == "dllp":
            model = DLLP(lr=0.01, n_epochs=100, hidden_layer_sizes=(1000,), n_jobs=0, random_state=seed[execution], device=device, model_type=model_type, pretrained=True)
        elif args.model == "mixbag":
            # Hyperparameters for mixbag (getting the best hyperparams from the paper)
            confidence_interval = 0.005 # 99% confidence interval
            choice = "uniform" # y-sampling method
            consistency = "vat" # add consistency loss (LLP-VAT + MixBag)
            # Hyperparameters for VAT (from their implementation, available at: https://github.com/kevinorjohn/LLP-VAT/blob/a111d6785e8b0b79761c4d68c5b96288048594d6/llp_vat/main.py#L360)
            # It is also used by LLPFC implementation of LLPVAT as default (https://github.com/Z-Jianxin/LLPFC/blob/main/utils.py#L167)
            xi = 1e-6
            eps = 6.0
            ip = 1
            model = MixBag(lr=0.01, n_epochs=100, hidden_layer_sizes=(1000,), n_jobs=0, random_state=seed[execution], device=device, model_type=model_type, pretrained=True, choice=choice, confidence_interval=confidence_interval, consistency=consistency, xi=xi, eps=eps, ip=ip)
        elif args.model == "llp-vat":
            # Hyperparameters for VAT (from their implementation, available at: https://github.com/kevinorjohn/LLP-VAT/blob/a111d6785e8b0b79761c4d68c5b96288048594d6/llp_vat/main.py#L360)
            xi = 1e-6
            eps = 6.0
            ip = 1
            model = LLPVAT(lr=0.01, n_epochs=100, hidden_layer_sizes=(1000,), n_jobs=0, random_state=seed[execution], device=device, model_type=model_type, pretrained=True, xi=xi, eps=eps, ip=ip)
        elif args.model == "llpfc":
            # Hyperparameters for LLPFC (using their default and the best noisy prior in the ResNet-18 experiments in the paper)
            batch_size = 128
            noisy_prior_choice = "approx"
            weights = "uniform"
            num_epoch_regroup = 20

            model = LLPFC(lr=0.01, n_epochs=100, model_type=model_type, device=device, pretrained=True, hidden_layer_sizes=(1000,), batch_size=batch_size, noisy_prior_choice=noisy_prior_choice, weights=weights, num_epoch_regroup=num_epoch_regroup, verbose=False, n_jobs=0, random_state=seed[execution])

        if args.model in NN_BASED_METHODS:
            # In the NN based methods, we use only one job (n_jobs=1)
            gs = gridSearchCVExperiments(model, params, refit=True, cv=args.n_splits, splitter=args.splitter, loss_type=args.loss, 
                            validation_size=args.validation_size, central_tendency_metric="mean", 
                            n_jobs=1, random_state=seed[execution])
        else:
            gs = gridSearchCVExperiments(model, params, refit=True, cv=args.n_splits, splitter=args.splitter, loss_type=args.loss, 
                            validation_size=args.validation_size, central_tendency_metric="mean", 
                            n_jobs=N_JOBS, random_state=seed[execution])
        
        metrics = ["abs", "oracle", "hypergeo"]

        n_classes = len(np.unique(y))
        average = "binary" if n_classes == 2 else "macro"

        if not sys.warnoptions:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"
            gs.fit(X_train, bags_train, proportions, y_train)
            for metric in metrics:
                print("Metric: %s" % metric)
                y_pred_train = gs.predict(X_train, metric)
                y_pred_test = gs.predict(X_test, metric)
                accuracy_train = accuracy_score(y_train, y_pred_train)
                accuracy_test = accuracy_score(y_test, y_pred_test)
                f1_train = f1_score(y_train, y_pred_train, average=average)
                f1_test = f1_score(y_test, y_pred_test, average=average)
                if metric == "abs":
                    best_hyperparams = gs.best_params_abs_
                elif metric == "oracle":
                    best_hyperparams = gs.best_params_oracle_
                elif metric == "hypergeo":
                    best_hyperparams = gs.best_params_hypergeo_

                df_results = pd.concat([df_results, pd.DataFrame([[metric, accuracy_train, accuracy_test, f1_train, f1_test, best_hyperparams]], columns=df_results.columns)], ignore_index=True)
        else:
            print("Warning failed!!!")

        df_results.to_parquet(filename)
        print("Execution finished!!!")
        print("Time: %s\n\n" % (time.time() - start))