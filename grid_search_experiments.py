from copy import deepcopy
import sys
import numpy as np
import pandas as pd
import warnings
from scipy.stats import hypergeom, binom
from sklearn.metrics import accuracy_score
from llp_learn.model_selection import gridSearchCV
import torch

class gridSearchCVExperiments(gridSearchCV):
    def _evaluate_candidate(self, X, bags, proportions, arg):
        est, param_id, param, train_index, validation_index = arg
        estimator = deepcopy(est)
        estimator.set_params(**param)
        try:
            estimator.fit(X[train_index], bags[train_index], proportions)
        except ValueError:
            return None
        y_pred_validation = estimator.predict(X[validation_index])

        predicted_proportions = np.empty(len(proportions))
        bag_size_validation = np.empty(len(proportions), int)
        num_bags = len(proportions)

        # Computing the predicted proportions and the size of bags in the validation set
        for i in range(num_bags):
            bag_validation = np.where(bags[validation_index] == i)[0]
            bag_size_validation[i] = len(bag_validation)
            y_pred_bag_validation = y_pred_validation[bag_validation]
            if len(bag_validation) == 0:
                predicted_proportions[i] = np.nan
            else:
                predicted_proportions[i] = np.count_nonzero(
                    y_pred_bag_validation == 1) / len(y_pred_bag_validation)

        # # Hypergeometric loss
        if self.splitter == "split-bag-shuffle":
            N = np.empty(num_bags, int)
            n = np.empty(num_bags, int)
            k = np.empty(num_bags, int)
            for i in range(num_bags):
                bag_train = np.where(bags[train_index] == i)[0]
                bag_validation = np.where(bags[validation_index] == i)[0]
                N[i] = len(bag_train) + len(bag_validation)
                n[i] = len(bag_validation)
                k[i] = np.count_nonzero(y_pred_validation[bag_validation] == 1)

            K = np.round(proportions * N).astype(int) # K items in each bag
            err_hypergeo = -hypergeom.logpmf(k, N, K, n)
            err_hypergeo[np.isinf(err_hypergeo)] = 10e5
            err_hypergeo = np.sum(err_hypergeo)
        elif self.splitter == "split-bag-bootstrap":
            n = np.empty(num_bags, int)
            k = np.empty(num_bags, int)
            for i in range(num_bags):
                bag_train = np.where(bags[train_index] == i)[0]
                bag_validation = np.where(bags[validation_index] == i)[0]
                n[i] = len(bag_train) + len(bag_validation)
                k[i] = np.count_nonzero(y_pred_validation[bag_validation] == 1)
            err_hypergeo = -binom.logpmf(k, n, proportions)
            err_hypergeo[np.isinf(err_hypergeo)] = 10e5
            err_hypergeo = np.sum(err_hypergeo)
        else:
            err_hypergeo = 0.0 # Not implemented

        err_abs = self._bag_loss(proportions, predicted_proportions).sum()

        err_oracle = 1 - \
            accuracy_score(self.y[validation_index], y_pred_validation)
        
        # Cleaning the GPU memory
        del estimator

        return (param_id, param, train_index, validation_index, err_abs, err_oracle, err_hypergeo)
    
    def _aggregate_results(self, r):
        df = pd.DataFrame(
            r, columns="id params train_index validation_index error_abs error_oracle error_hypergeo".split())

        # Removing hyperparameters that do not converged in all folds
        df = df.groupby("id").filter(lambda x: len(x) == self.cv)

        if self.central_tendency_metric == "mean":
            df_results = df["id error_abs error_oracle error_hypergeo".split()].groupby("id").mean()
        elif self.central_tendency_metric == "median":
            df_results = df["id error_abs error_oracle error_hypergeo".split()].groupby("id").median()
        else:
            raise Exception(
                "There was not possible to computate the error. Verify the central_tendency_metric parameter.")
        
        return df, df_results
    
    def _fit_best_estimator(self, X, bags, proportions, df, df_results):
        # Metric 1) error abs
        best_estimator_id = int(df_results["error_abs"].idxmin())
        df_best_estimator = df[df.id == best_estimator_id]
        self.best_params_abs_ = df_best_estimator["params"].iloc[0]

        if self.refit:
            self.best_estimator_abs_ = deepcopy(self.estimator)
            # Release GPU memory
            torch.cuda.empty_cache()
            self.best_estimator_abs_.set_params(**self.best_params_abs_)
            try:
                self.best_estimator_abs_.fit(X, bags, proportions)
            except ValueError:
                self.best_estimator_abs_ = None
                warnings.warn("Error abs case: The best hyperparameters found by the CV process did not converge in the refit process. \
                    The best hyperparameters are " + str(self.best_params_abs_))
        else:
            df_best_estimator_abs = df_best_estimator[df_best_estimator.error_abs == df_best_estimator.error_abs.min()]
            train_index = df_best_estimator_abs["train_index"].iloc[0]
            # Fit the best_estimator_abs_
            self.best_estimator_abs_ = deepcopy(self.estimator)
            # Release GPU memory
            torch.cuda.empty_cache()
            self.best_estimator_abs_.set_params(**self.best_params_abs_)
            self.best_estimator_abs_.fit(X[train_index], bags[train_index], proportions)
        
        # Saving the variables used to refit later
        self.X_ = X
        self.bags_ = bags
        self.proportions_ = proportions

        # Metric 2) oracle
        best_estimator_id = int(df_results["error_oracle"].idxmin())
        df_best_estimator = df[df.id == best_estimator_id]
        self.best_params_oracle_ = df_best_estimator["params"].iloc[0]

        if self.refit:
            self.best_estimator_oracle_ = deepcopy(self.estimator)
            self.best_estimator_oracle_.set_params(**self.best_params_oracle_)
            try:
                self.best_estimator_oracle_.fit(X, bags, proportions)
            except ValueError:
                self.best_estimator_oracle_ = None
                warnings.warn("Error abs case: The best hyperparameters found by the CV process did not converge in the refit process. \
                    The best hyperparameters are " + str(self.best_params_oracle_))
        else:
            df_best_estimator_oracle = df_best_estimator[df_best_estimator.error_oracle == df_best_estimator.error_oracle.min()]
            train_index = df_best_estimator_oracle["train_index"].iloc[0]
            # Fit the best_estimator_oracle_
            self.best_estimator_oracle_ = deepcopy(self.estimator)
            # Release GPU memory
            torch.cuda.empty_cache()
            self.best_estimator_oracle_.set_params(**self.best_params_oracle_)
            self.best_estimator_oracle_.fit(X[train_index], bags[train_index], proportions)

        # # Metric 3) hypergeo
        best_estimator_id = int(df_results["error_hypergeo"].idxmin())
        df_best_estimator = df[df.id == best_estimator_id]
        self.best_params_hypergeo_ = df_best_estimator["params"].iloc[0]

        if self.refit:
            self.best_estimator_hypergeo_ = deepcopy(self.estimator)
            self.best_estimator_hypergeo_.set_params(**self.best_params_hypergeo_)
            try:
                self.best_estimator_hypergeo_.fit(X, bags, proportions)
            except ValueError:
                self.best_estimator_hypergeo_ = None
                warnings.warn("Error abs case: The best hyperparameters found by the CV process did not converge in the refit process. \
                    The best hyperparameters are " + str(self.best_params_hypergeo_))
        else:
            df_best_estimator_hypergeo = df_best_estimator[df_best_estimator.error_hypergeo == df_best_estimator.error_hypergeo.min()]
            train_index = df_best_estimator_hypergeo["train_index"].iloc[0]
            # Fit the best_estimator_hypergeo_
            self.best_estimator_hypergeo_ = deepcopy(self.estimator)
            # Release GPU memory
            torch.cuda.empty_cache()
            self.best_estimator_hypergeo_.set_params(**self.best_params_hypergeo_)
            self.best_estimator_hypergeo_.fit(X[train_index], bags[train_index], proportions)
    
    def predict(self, X, metric):
        if metric == "abs":
            return self.best_estimator_abs_.predict(X)
        elif metric == "oracle":
            return self.best_estimator_oracle_.predict(X)
        elif metric == "hypergeo":
            return self.best_estimator_hypergeo_.predict(X)
        else:
            return ValueError("metric %s is not valid" % metric)    