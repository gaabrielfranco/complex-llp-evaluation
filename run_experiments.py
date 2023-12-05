# os.environ["OMP_NUM_THREADS"] = str(N_JOBS) # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS) # export OPENBLAS_NUM_THREADS=4 
# os.environ["MKL_NUM_THREADS"] = str(N_JOBS) # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_JOBS) # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS) # export NUMEXPR_NUM_THREADS=6

import argparse
import os
import sys
import pandas as pd
import time
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from llp_learn.em import EM
from llp_learn.dllp import DLLP
from llp_learn.util import compute_proportions
from llp_learn.mixbag import MixBag
from llp_learn.llpvat import LLPVAT

from grid_search_experiments import gridSearchCVExperiments
from almostnolabel import MM, LMM, AMM

import torchvision.transforms as transforms
import torch

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
    n_executions = 5 # Number of executions (it was 30 before, maybe will increase in the future)

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
        "dllp", "mixbag", "llp-vat"
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device: %s" % device)

    directory = "llp-benchmark-results/"

    # Parsing arguments
    parser = argparse.ArgumentParser(description="LLP benchmark experiments")
    parser.add_argument("--dataset", "-d", required=True, help="the dataset that will be used in the experiments")
    parser.add_argument("--model", "-m", choices=["kdd-lr", "lmm", "amm", "mm", "dllp", "mixbag"], required=True,
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
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
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
                f1_train = f1_score(y_train, y_pred_train)
                f1_test = f1_score(y_test, y_pred_test)
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