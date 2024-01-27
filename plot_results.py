import argparse
from copy import deepcopy
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
from scipy.stats import ttest_ind
from llp_learn.util import compute_proportions

def get_dataset_variant(dataset):
    if "naive" in dataset:
         return "Naive"
    elif "simple" in dataset:
        return "Simple"
    elif "intermediate" in dataset:
        return "Intermediate"
    elif "hard" in dataset:
        return "Hard"
    else:
        return "unknown"

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--plot_type', "-p", type=str, required=True, help="Plot to generate")
parser.add_argument('--n_classes', "-n", choices=["binary", "multiclass"], type=str, required=True, help="Number of classes")
parser.add_argument("--split_domain", "-s", action="store_true", help="Split the domain in the plots")
args = parser.parse_args()

VARIANTS = ["naive", "simple", "intermediate", "hard"]

dataset_title_map = {}
error_legend_map = {
    "error_bag_abs": "Abs loss",
     "error_accuracy_validation": "Oracle"
}

"""
Attention:
    This can change depending on the experiments - e.g. n_folds and validation perc size
"""
split_method_map = {
    'split-bag-bootstrap': "SB\nBS",
    'split-bag-shuffle': "SB\nSH",
    'split-bag-k-fold': "SB\nKF",
    'full-bag-stratified-k-fold': "FB\nKF",
}

error_metric_map = {
    "error_bag_abs": "Abs loss",
}

final_results = pd.read_parquet("llp-benchmark-experiment-results.parquet")
# Getting only abs metric
final_results = final_results[final_results.metric == "abs"]

final_results.rename(columns={"metric": "error_metric"}, inplace=True)

final_results["error_metric"].replace(error_legend_map, inplace=True)
final_results["split_method"].replace(split_method_map, inplace=True)

final_results["split_method"] = final_results["split_method"] + "\n" + final_results["validation_size_perc"].astype(str)
final_results["split_method"] = final_results["split_method"].str.replace("nan", "")

final_results["error_metric"].replace(error_metric_map, inplace=True)

base_datasets = ["cifar-10", "adult", "cifar-10-grey", "svhn"]

base_datasets_supervised_accuracy = {
    "cifar-10": 0.5105,
    "adult": 0.8414,
    "cifar-10-grey": 0.9279,
    "svhn": 0.9369,
}

base_datasets_supervised_f1_score = {
    "cifar-10": 0.5116,
    "adult": 0.6358,
    "cifar-10-grey": 0.9403,
    "svhn": 0.9325,
}

base_datasets_type = {
    "cifar-10-grey": "Image-Objects",
    "adult": "Tabular"
}

model_map = {
    "lmm": "LMM",
    "llp-svm-lin": "Alter-SVM",
    "kdd-lr": "EM/LR",
    "mm": "MM",
    "dllp": "DLLP",
    "amm": "AMM",
    "mixbag": "MixBag + LLP-VAT",
    "llp-vat": "LLP-VAT",
    "llpfc": "LLPFC",
}

# Getting the infos about the datasets
final_results["n_bags"] = final_results.dataset.apply(lambda x: "extra-extra-large" if "extra-extra-large" in x else "extra-large" if "extra-large" in x else "large" if "large" in x else "small" if "small" in x else "massive" if "massive" in x else "none")
final_results["bag_sizes"] = final_results.dataset.apply(lambda x: "not-equal" if "not-equal" in x else "equal" if "equal" in x else "fol-clust" if "fol-clust" in x else "none")
final_results["proportions"] = final_results.dataset.apply(lambda x: "close-global" if "close-global" in x else "far-global" if "far-global" in x else "mixed" if "mixed" in x else "fol-clust" if "fol-clust" in x else "none")

final_results["model"].replace(model_map, inplace=True)

# Creating a column with the dataset variant
final_results["dataset_variant"] = final_results["dataset"].apply(get_dataset_variant)

# Correcting the proportions for the naive variant
final_results.loc[final_results.dataset_variant == "Naive", "proportions"] = "none"

# Creating a columns with the base dataset
final_results["base_dataset"] = "None"
for dataset in base_datasets:
    final_results.loc[final_results.dataset.str.contains(dataset), "base_dataset"] = dataset

final_results["dataset_type"] = "None"
for base_dataset in base_datasets_type:
    final_results.loc[final_results.base_dataset.str.contains(base_dataset), "dataset_type"] = base_datasets_type[base_dataset]

if args.n_classes == "binary":
    final_results = final_results[((final_results.base_dataset != "cifar-10") & (final_results.base_dataset != "svhn"))]
else:
    final_results = final_results[((final_results.base_dataset == "cifar-10") | (final_results.base_dataset == "svhn"))]

# Getting all models
all_models = final_results["model"].unique()

# Adding domain column
if args.split_domain:
    final_results["dataset_domain"] = final_results["bag_sizes"].apply(lambda x: "large" if x == "fol-clust" else "small")

# We have a total of 96 datasets for binary classification
# We have a total of 8 algoritms for binary classification
# We have a total of 5 execs for binary classification (we will increase it to 10)
# Total = 96 * 8 * 5 = 3840
if args.plot_type == "check-n-experiments":
    total_models = len(final_results)
    print("Total trained models: ", total_models)
    for model in final_results["model"].unique():
        print(model, len(final_results[final_results["model"] == model]))
    print("")
    # Checking number of experiments
    n_experiments_df = final_results.groupby(["model", "dataset", "split_method"]).size().reset_index(name='counts').sort_values(by="counts", ascending=False)
    print("Total number of experiments:", len(n_experiments_df))
    print("Experiments per split_method")
    print(n_experiments_df["split_method"].value_counts())
elif args.plot_type == "datasets-info":
    dataset_info = pd.DataFrame(columns=["Dataset", "Number of bags", "Proportions", "Bag sizes"])

    files = glob.glob("datasets-ci/*.parquet")
    files = sorted(files)

    # Removing base datasets
    files = [file for file in files if "adult.parquet" not in file and "cifar-10-grey-animal-vehicle.parquet" not in file]

    for file in files:

        dataset = file.split("/")[-1].split(".")[0]

        # Extracting variant
        llp_variant = [variant for variant in VARIANTS if variant in dataset][0]

        # Extracting base dataset
        base_dataset = dataset.split(llp_variant)[0]
        base_dataset = base_dataset[:-1]

        # Reading X, y (base dataset) and bags (dataset)
        df = pd.read_parquet("{}/{}.parquet".format("datasets-ci", base_dataset))
        X = df.drop(["y"], axis=1).values
        y = df["y"].values.reshape(-1, 1)

        df = pd.read_parquet(file)
        bags = df["bag"].values

        proportions = compute_proportions(bags, y)
        proportions = [round(x, 2) for x in proportions]
        bags_sizes = np.bincount(bags)
        list2str = lambda x: ("(" + ", ".join([str(y) for y in x]) + ")").replace(",)", ")")
        dataset_info = pd.concat([dataset_info, pd.DataFrame({"Dataset": [dataset], "Number of bags": [len(np.unique(bags))], "Proportions": [list2str(proportions)], "Bag sizes": [list2str(bags_sizes)]})], ignore_index=True)
    dataset_info.sort_values(by=["Dataset"], inplace=True)
    with pd.option_context("max_colwidth", 10000):
        dataset_info.to_latex(buf="tables/table-datasets-info.tex", index=False, escape=False, longtable=True)
elif args.plot_type == "best-methods":
    if args.n_classes == "binary":
        hue_order = ["MM", "AMM", "LMM", "EM/LR", "DLLP", "LLP-VAT", "MixBag + LLP-VAT", "LLPFC"]
    else:
        hue_order = ["DLLP", "LLP-VAT", "MixBag + LLP-VAT", "LLPFC"]
    # Plot mean accuracy of each method per dataset
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rc('font', size=6)
    g = sns.catplot(y="base_dataset", x="accuracy_test", hue="model", col="dataset_variant", data=final_results, kind="bar", col_order=["Hard", "Intermediate", "Simple", "Naive"], legend=False, height=2, aspect=1.5, sharex=True, errorbar="sd", capsize=0.05, col_wrap=2, errwidth=1, hue_order=hue_order)
    # Draw a line with the accuracy of the supervised neural network
    for ax in g.axes.flat:
        if args.n_classes == "binary":
            ax.axvline(x=base_datasets_supervised_accuracy["adult"], ymin=0.5, ymax=1, color="black", linestyle="--", label="Adult supervised NN") # Adult performance
            ax.axvline(x=base_datasets_supervised_accuracy["cifar-10-grey"], ymin=0, ymax=0.5, color="red", linestyle="--", label="CIFAR-10-Grey supervised NN") # CIFAR-10-Grey performance
        else:
            ax.axvline(x=base_datasets_supervised_accuracy["cifar-10"], ymin=0.5, ymax=1, color="black", linestyle="--", label="CIFAR-10 supervised NN") # CIFAR-10 performance
            ax.axvline(x=base_datasets_supervised_accuracy["svhn"], ymin=0, ymax=0.5, color="red", linestyle="--", label="SVHN supervised NN") # SVHN performance

    # Say that the line is the accuracy of the supervised neural network in the legend
    plt.legend(loc="best", borderaxespad=0., fontsize=5)
    g.set_xlabels("Accuracy")
    g.set_ylabels("Base Dataset")
    plt.tight_layout()
    filename = f"plots/{args.n_classes}-avg-performance-per-method-accuracy.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rc('font', size=6)
    g = sns.catplot(y="n_bags", x="accuracy_test", hue="model", col="base_dataset", data=final_results, kind="bar", legend=False, height=2, aspect=1.5, sharex=True, errorbar="sd", capsize=0.05, col_wrap=2, errwidth=1, hue_order=hue_order)
    # Draw a line with the accuracy of the supervised neural network
    for i, ax in enumerate(g.axes.flat):
        if i == 0:
            if args.n_classes == "binary":
                ax.axvline(x=base_datasets_supervised_accuracy["adult"], color="black", linestyle="--", label="Adult supervised NN") # Adult performance
            else:
                ax.axvline(x=base_datasets_supervised_accuracy["cifar-10"], color="black", linestyle="--", label="CIFAR-10 supervised NN") # CIFAR-10 performance
        else:
            if args.n_classes == "binary":
                ax.axvline(x=base_datasets_supervised_accuracy["cifar-10-grey"], color="red", linestyle="--", label="CIFAR-10-Grey supervised NN") # CIFAR-10-Grey performance
            else:
                ax.axvline(x=base_datasets_supervised_accuracy["svhn"], color="red", linestyle="--", label="SVHN supervised NN") # SVHN performance
            

    # Say that the line is the accuracy of the supervised neural network in the legend
    plt.legend(loc="best", borderaxespad=0., fontsize=5)
    g.set_xlabels("Accuracy")
    g.set_ylabels("Number of bags")
    plt.tight_layout()
    filename = f"plots/{args.n_classes}-avg-performance-per-n_bags-accuracy.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # f1-score
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rc('font', size=6)
    g = sns.catplot(y="base_dataset", x="f1_test", hue="model", col="dataset_variant", data=final_results, kind="bar", col_order=["Hard", "Intermediate", "Simple", "Naive"], legend=False, height=2, aspect=1.5, sharex=True, errorbar="sd", capsize=0.05, col_wrap=2, errwidth=1, hue_order=hue_order)
    # Draw a line with the f-score of the supervised neural network
    for ax in g.axes.flat:
        if args.n_classes == "binary":
            ax.axvline(x=base_datasets_supervised_f1_score["adult"], ymin=0.5, ymax=1, color="black", linestyle="--", label="Adult supervised NN") # Adult performance
            ax.axvline(x=base_datasets_supervised_f1_score["cifar-10-grey"], ymin=0, ymax=0.5, color="red", linestyle="--", label="CIFAR-10-Grey supervised NN") # CIFAR-10-Grey performance
        else:
            ax.axvline(x=base_datasets_supervised_f1_score["cifar-10"], ymin=0.5, ymax=1, color="black", linestyle="--", label="CIFAR-10 supervised NN") # CIFAR-10 performance
            ax.axvline(x=base_datasets_supervised_f1_score["svhn"], ymin=0, ymax=0.5, color="red", linestyle="--", label="SVHN supervised NN") # SVHN performance

    # Say that the line is the accuracy of the supervised neural network in the legend
    plt.legend(loc="best", borderaxespad=0., fontsize=5)
    g.set_xlabels(r"$F_1$-score")
    g.set_ylabels("Base Dataset")
    plt.tight_layout()
    filename = f"plots/{args.n_classes}-avg-performance-per-method-f1.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rc('font', size=6)
    g = sns.catplot(y="n_bags", x="f1_test", hue="model", col="base_dataset", data=final_results, kind="bar", legend=False, height=2, aspect=1.5, sharex=True, errorbar="sd", capsize=0.05, col_wrap=2, errwidth=1, hue_order=hue_order)
    # Draw a line with the f-score of the supervised neural network
    for i, ax in enumerate(g.axes.flat):
        if i == 0:
            if args.n_classes == "binary":
                ax.axvline(x=base_datasets_supervised_f1_score["adult"], color="black", linestyle="--", label="Adult supervised NN") # Adult performance
            else:
                ax.axvline(x=base_datasets_supervised_f1_score["cifar-10"], color="black", linestyle="--", label="CIFAR-10 supervised NN") # CIFAR-10 performance
        else:
            if args.n_classes == "binary":
                ax.axvline(x=base_datasets_supervised_f1_score["cifar-10-grey"], color="red", linestyle="--", label="CIFAR-10-Grey supervised NN") # CIFAR-10-Grey performance
            else:
                ax.axvline(x=base_datasets_supervised_f1_score["svhn"], color="red", linestyle="--", label="SVHN supervised NN") # SVHN performance

    # Say that the line is the accuracy of the supervised neural network in the legend
    plt.legend(loc="best", borderaxespad=0., fontsize=5)
    g.set_xlabels(r"$F_1$-score")
    g.set_ylabels("Number of bags")
    plt.tight_layout()
    filename = f"plots/{args.n_classes}-avg-performance-per-n_bags-f1.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    df_best_methods = pd.DataFrame(columns=["base_dataset", "dataset_variant", "n_bags", "bag_sizes", "proportions", "best_hyperparam_method", "best_algorithm", "best_in_both"])
    diff_best_model_bottom = {}
    for base_dataset in sorted(final_results.base_dataset.unique()):
        for llp_variant in sorted(final_results.dataset_variant.unique()):
            for n_bags in sorted(final_results.n_bags.unique()):
                for bag_sizes in sorted(final_results.bag_sizes.unique()):
                    for proportions in sorted(final_results.proportions.unique()):
                        if proportions == "none" and llp_variant != "Naive":
                            continue # Skip the none proportion (naive case)
                        
                        best_method = deepcopy(final_results[(final_results.base_dataset == base_dataset) & (final_results.dataset_variant == llp_variant) & (final_results.n_bags == n_bags) & (final_results.bag_sizes == bag_sizes) & (final_results.proportions == proportions)])

                        # Combination doesn't exist (case of CIFAR-10 intermediate that are not close-global)
                        if best_method.shape[0] == 0:
                            continue
                        
                        # Removing the \n from the split method (used to make the table more readable)
                        best_method["split_method"] = best_method.split_method.apply(lambda x: x.replace("\n", " "))
                        best_method["split_method"] = best_method.split_method.apply(lambda x: x.replace("KF ", "KF"))

                        # Get the overall best method
                        x = best_method.groupby(["split_method", "model"]).mean(numeric_only=True).f1_test.sort_values(ascending=False)
                        best_global_combination = set()

                        # The top (split_method, model) combination is always included in the best global
                        best_global_combination.add((x.index[0][0], x.index[0][1]))
                        for i in range(1, len(x.index)):
                            split_method_1, model_1 = x.index[0] 
                            split_method_2, model_2 = x.index[i]

                            acc_1 = best_method[(best_method.split_method == split_method_1) & (best_method.model == model_1)].f1_test.values
                            acc_2 = best_method[(best_method.split_method == split_method_2) & (best_method.model == model_2)].f1_test.values

                            best_models_test = ttest_ind(acc_1, acc_2, equal_var=False, random_state=73921)
                            if best_models_test.pvalue <= 0.05:
                                # The top (split_method, model) is better than the i-th (split_method, model) combination
                                break
                            else:
                                # split_method_1, model_1 are already in the best global combination.
                                # Then, add split_method_2, model_2
                                best_global_combination.add((split_method_2, model_2))

                        # Get the best model (algorithm) for this combination of parameters
                        accuracy_models = {}
                        avg_accuracy_models = {}
                        for model in best_method.model.unique():
                            accuracy_models[model] = deepcopy(best_method[best_method.model == model].f1_test.values)
                            avg_accuracy_models[model] = np.mean(accuracy_models[model])

                        avg_accuracy_models = sorted(avg_accuracy_models.items(), key=lambda x: x[1], reverse=True)
                        best_models = set()

                        # The top model is always be in the best model set
                        best_models.add(avg_accuracy_models[0][0])

                        for i in range(1, len(avg_accuracy_models)):
                            best_models_test = ttest_ind(accuracy_models[avg_accuracy_models[0][0]],
                                accuracy_models[avg_accuracy_models[i][0]],
                                equal_var=False, random_state=73921)
                            if best_models_test.pvalue <= 0.05:
                                # Computing the difference:
                                # worst method of the set of best methods - method right below the set of best methods
                                try:
                                    diff_best_model_bottom[llp_variant].append(avg_accuracy_models[i-1][1] - avg_accuracy_models[i][1])
                                except KeyError:
                                    diff_best_model_bottom[llp_variant] = [avg_accuracy_models[i-1][1] - avg_accuracy_models[i][1]]
                                break
                            else:
                                best_models.add(avg_accuracy_models[i][0])

                        # Get the best hyperparameter method
                        # Each model "votes" for the best hyperparameter method
                        best_split_method_votes = {}
                        for split_method in best_method.split_method.unique():
                            best_split_method_votes[split_method] = 0

                        for model in best_method.model.unique():
                            accuracy_split_method = {}
                            avg_accuracy_split_method = {}
                            # Get the best hyperparameter method for this model
                            best_method_model = best_method[best_method.model == model]
                            for split_method in best_method_model.split_method.unique():
                                accuracy_split_method[split_method] = deepcopy(best_method_model[best_method_model.split_method == split_method].f1_test.values)
                                avg_accuracy_split_method[split_method] = np.mean(accuracy_split_method[split_method])

                            avg_accuracy_split_method = sorted(avg_accuracy_split_method.items(), key=lambda x: x[1], reverse=True)
                            best_split_method = set()

                            # The top split method is always be in the best split method set
                            best_split_method.add(avg_accuracy_split_method[0][0])

                            for i in range(1, len(avg_accuracy_split_method)):
                                best_split_method_test = ttest_ind(accuracy_split_method[avg_accuracy_split_method[0][0]],
                                    accuracy_split_method[avg_accuracy_split_method[i][0]],
                                    equal_var=False, random_state=73921)
                                if best_split_method_test.pvalue <= 0.05:
                                    break
                                else:
                                    best_split_method.add(avg_accuracy_split_method[i][0])

                            for split_method in best_split_method:
                                best_split_method_votes[split_method] += 1

                        # Get the best hyperparameter method
                        best_split_method_votes = sorted(best_split_method_votes.items(), key=lambda x: x[1], reverse=True)
                        
                        # Getting the split methods with max values from the tuple (split_method, votes)
                        max_votes = best_split_method_votes[0][1]
                        best_split_method = set()
                        for split_method, votes in best_split_method_votes:
                            if votes == max_votes:
                                best_split_method.add(split_method)
                            else:
                                break

                        df_best_methods = pd.concat([df_best_methods, pd.DataFrame({
                           "base_dataset": base_dataset,
                           "dataset_variant": llp_variant,
                           "n_bags": n_bags,
                           "bag_sizes": bag_sizes,
                           "proportions": proportions,
                           "best_hyperparam_method": str(sorted(best_split_method)),
                           "best_algorithm": str(sorted(best_models)),
                           "best_in_both": str(sorted(best_global_combination))
                        }, index=[0])], ignore_index=True)
    
    # Categorizing the best_hyperparam_method 
    def get_best_hyperparam_method_cat(x):
        if "SB" in x and "FB" in x:
            return "SB+FB"
        elif "SB" in x:
            return "SB"
        elif "FB" in x:
            return "FB"
        
    df_best_methods["best_hyperparam_method_cat"] = df_best_methods.best_hyperparam_method.apply(get_best_hyperparam_method_cat)

    # # # Getting only large domains
    # # df_best_methods = df_best_methods[~df_best_methods.n_bags.isin(["extra-large", "extra-extra-large", "massive"])]

    # # Heatmap using Generalized Jaccard Index

    # Computing the Jaccard Index (multiset version)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rc('font', size=6)
    #fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True, sharex=True)
    matrices_jaccard = {}
    for idx, base_dataset in enumerate(df_best_methods.base_dataset.unique()):
        x = df_best_methods[df_best_methods.base_dataset == base_dataset]
        matrix_jaccard = np.zeros((4, 4), dtype=np.float32)
        for i, dataset_variant_1 in enumerate(["Naive", "Simple", "Intermediate", "Hard"]):
            for j, dataset_variant_2 in enumerate(["Naive", "Simple", "Intermediate", "Hard"]):
                x1 = x[(x.dataset_variant == dataset_variant_1)].best_algorithm.values
                x2 = x[(x.dataset_variant == dataset_variant_2)].best_algorithm.values

                multiset_x1 = []
                for ba in x1:
                    for elem in eval(ba):
                        multiset_x1.append(elem)
                
                multiset_x2 = []
                for ba in x2:
                    for elem in eval(ba):
                        multiset_x2.append(elem)
                
                num = 0
                den = 0
                for alg in all_models:
                    count_alg_x1 = multiset_x1.count(alg)
                    count_alg_x2 = multiset_x2.count(alg)
                    num += min(count_alg_x1, count_alg_x2)
                    den += max(count_alg_x1, count_alg_x2)

                # Computing the Generalized Jaccard Index
                matrix_jaccard[i, j] = num / den

        matrices_jaccard[base_dataset] = matrix_jaccard
        # Plotting the heatmap
        #sns.heatmap(matrix_jaccard, annot=True, cmap="YlGnBu", xticklabels=["Naive", "Simple", "Intermediate", "Hard"], yticklabels=["Naive", "Simple", "Intermediate", "Hard"], ax=ax[idx], vmin=0, vmax=1, annot_kws={"size": 5})
        
        # ax[idx].set_xlabel("Dataset Variant")
        # ax[idx].set_ylabel("Dataset Variant")
        # ax[idx].set_title(base_dataset)
    fig = plt.figure()
    gs0 = matplotlib.gridspec.GridSpec(1,2, width_ratios=[20,2], hspace=0.05)
    gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs0[1], hspace=0, wspace=1.5)

    ax = fig.add_subplot(gs0[0])
    cax1 = fig.add_subplot(gs00[0])
    cax2 = fig.add_subplot(gs00[1])
    
    vmin, vmax = 0, 1
    from matplotlib.colors import ListedColormap
    for i, base_dataset in enumerate(df_best_methods.base_dataset.unique()):
        print(base_dataset)
        matrix_jaccard = matrices_jaccard[base_dataset]
        if i == 1:
            mask = np.zeros_like(matrix_jaccard, dtype=bool)
             # Fill above diagonal with True
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(matrix_jaccard, annot=True, mask=mask, cmap='Blues', vmin=vmin, vmax=vmax, ax=ax, cbar_ax=cax2, cbar_kws={"label": base_dataset}, xticklabels=["Naive", "Simple", "Intermediate", "Hard"], yticklabels=["Naive", "Simple", "Intermediate", "Hard"], annot_kws={"size": 5})
        else:
            mask = np.zeros_like(matrix_jaccard, dtype=bool)
             # Fill bellow diagonal with True
            mask[np.tril_indices_from(mask)] = True
            sns.heatmap(matrix_jaccard, annot=True, mask=mask, cmap='OrRd', vmin=vmin, vmax=vmax, ax=ax, cbar_ax=cax1, cbar_kws={"label": base_dataset, "ticks":[]}, xticklabels=["Naive", "Simple", "Intermediate", "Hard"], yticklabels=["Naive", "Simple", "Intermediate", "Hard"], annot_kws={"size": 5})
    sns.heatmap(np.ones((4, 4), dtype=int), mask=~np.eye(4, dtype=bool), cmap=ListedColormap(['white']), annot=True, annot_kws={"size": 5}, cbar=False, ax=ax, xticklabels=["Naive", "Simple", "Intermediate", "Hard"], yticklabels=["Naive", "Simple", "Intermediate", "Hard"])
    plt.tight_layout()
    filename = f"plots/{args.n_classes}-jaccard-index-heatmap-best-algorithm.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best hyperparams per base dataset and dataset variant
    D = df_best_methods.groupby(["base_dataset", "dataset_variant"]).best_hyperparam_method_cat.value_counts()
    # Normalize the count
    D = D.groupby(level=[0,1],group_keys=False).apply(lambda x: x/float(x.sum()))
    D = D.reset_index(name="count")
    D.rename(columns={"best_hyperparam_method_cat": "Best Hyperparam. Selection Method",
                      "base_dataset": "Base Dataset",
                      "dataset_variant": "Dataset Variant",
                      "count": "Proportion"}, inplace=True)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rc('font', size=6)

    g = sns.catplot(row="Base Dataset", y="Proportion", x="Best Hyperparam. Selection Method", col="Dataset Variant", data=D, kind="bar", errorbar=None, col_order=["Hard", "Intermediate", "Simple", "Naive"], legend=False, height=1.1, aspect=1.1, sharex=True)
    g.set_titles("{row_name}\n{col_name}")
    g.set_xlabels("")
    plt.tight_layout()
    filename = f"plots/{args.n_classes}-best-hyperparam-methods-per-base-dataset-and-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Print best algorithm per base dataset and dataset variant
    D = df_best_methods.groupby(["base_dataset", "dataset_variant"]).best_algorithm.value_counts()
    # Normalize the count
    D = D.groupby(level=[0,1],group_keys=False).apply(lambda x: x/float(x.sum()))
    D = D.reset_index(name="count")
    D["best_algorithm"] = D.best_algorithm.apply(lambda x: x.translate({ord("["): "", ord("]"): "", ord("'"): "", ord(" "): ""}))
    ba = D["best_algorithm"].unique()
    D["best_algorithm_legend"] = D["best_algorithm"].apply(lambda x: np.where(ba == x)[0][0])
    D.rename(columns={"best_algorithm": "Best Algorithm(s)",
                      "base_dataset": "Base Dataset",
                      "dataset_variant": "Dataset Variant",
                      "best_algorithm_legend": "Best Algorithm Index",
                      "count": "Proportion"}, inplace=True)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rc('font', size=6)

    g = sns.catplot(row="Base Dataset", y="Proportion", x="Best Algorithm Index", hue="Best Algorithm(s)", col="Dataset Variant", data=D, kind="bar", errorbar=None, col_order=["Hard", "Intermediate", "Simple", "Naive"], legend=False, height=1.1, aspect=1.1, sharex=True, dodge=False)#, width=1)
    g.set_titles("{row_name}\n{col_name}")
    g.set_xlabels("")
    g.set_xticklabels("")
    for ax in g.axes.flat:
        ax.set_xticks([])
    plt.legend(bbox_to_anchor=(1.1, 1.8), loc=2, borderaxespad=0., fontsize=5)
    plt.tight_layout()
    filename = f"plots/{args.n_classes}-best-algorithms-per-base-dataset-and-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Plot the effect size: how much the best algorithms are the best
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('font', size=6)
    
    fig, ax = plt.subplots(2, 2, figsize=(3.5, 2), sharey=True, sharex=True)
    for idx, llp_variant in enumerate(["Naive", "Simple", "Intermediate", "Hard"]):
        sns.histplot(diff_best_model_bottom[llp_variant], kde=True, ax=ax[idx // 2, idx % 2], kde_kws={'bw_adjust': 0.5}, stat="count")
        ax[idx // 2, idx % 2].set_xlabel("Difference in " + r"$F_1$" + "-score")
        ax[idx // 2, idx % 2].set_ylabel("Count")
        ax[idx // 2, idx % 2].set_title(llp_variant)
    plt.tight_layout()
    filename = f"plots/{args.n_classes}-effect-sizes-dist.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Plot the count of the best algorithm in total
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('font', size=6)

    D = df_best_methods.best_algorithm.value_counts()
    D = D.reset_index(name="count")
    D.rename(columns={"index": "best_algorithm"}, inplace=True)

    best_methods_count = {}
    # iterate over rows with iterrows()
    for index, row in D.iterrows():
        alg_list = eval(row["best_algorithm"])
        for alg in alg_list:
            try:
                best_methods_count[alg] += row["count"]
            except KeyError:
                best_methods_count[alg] = row["count"]

    D = pd.DataFrame.from_dict(best_methods_count, orient='index', columns=["count"])
    D = D.reset_index()
    D.rename(columns={"index": "Best Algorithm"}, inplace=True)
    D = D.sort_values(by="count", ascending=False)
    # Replace count with proportion
    D["count"] = D["count"] / D["count"].sum()
    D = D.sort_values(by="count", ascending=False)
    D.rename(columns={"count": "Proportion"}, inplace=True)
    g = sns.catplot(y="Best Algorithm", x="Proportion", data=D, kind="bar", errorbar=None, legend=False, height=2, aspect=1.5)
    plt.xlabel("Proportion")
    plt.ylabel("Best Algorithm")
    plt.tight_layout()
    filename = f"plots/{args.n_classes}-best-algorithms-count.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Plot the count of the best algorithm per base dataset
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('font', size=6)

    D = df_best_methods.groupby(["base_dataset"]).best_algorithm.value_counts()
    D = D.reset_index(name="count")
    D.rename(columns={"index": "best_algorithm"}, inplace=True)

    best_methods_count = {}
    for base_dataset in D.base_dataset.unique():
        best_methods_count[base_dataset] = {}
    
    # iterate over rows with iterrows()
    for index, row in D.iterrows():
        alg_list = eval(row["best_algorithm"])
        for alg in alg_list:
            try:
                best_methods_count[row["base_dataset"]][alg] += row["count"]
            except KeyError:
                best_methods_count[row["base_dataset"]][alg] = row["count"]

    D = pd.DataFrame.from_dict(best_methods_count, orient='index')
    D = D.reset_index()
    D.rename(columns={"index": "Base Dataset"}, inplace=True)
    D = D.sort_values(by="Base Dataset", ascending=False)
    D = D.melt(id_vars=["Base Dataset"], var_name="Best Algorithm", value_name="Count")
    D.dropna(inplace=True) # Removing algorithms that are not in the base dataset
    # Replace count with proportion
    D["Count"] = D["Count"] / D.groupby(["Base Dataset"])["Count"].transform('sum')
    D = D.sort_values(by=["Base Dataset", "Count"], ascending=False)
    D.rename(columns={"Count": "Proportion"}, inplace=True)

    g = sns.catplot(y="Base Dataset", x="Proportion", hue="Best Algorithm", data=D, kind="bar", errorbar=None, legend=False, height=2, aspect=1.5, palette="husl", hue_order=hue_order)
    plt.legend(loc="best", borderaxespad=0., fontsize=5)
    # xticks set
    g.set(xticks=[0, 0.2, 0.4, 0.6])
    g.set_xticklabels(["0", "0.2", "0.4", "0.6"])
    plt.xlabel("Proportion")
    plt.ylabel("Base Dataset")
    plt.tight_layout()
    filename = f"plots/{args.n_classes}-best-algorithms-count-per-base-dataset.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Plot the count of the best algorithm per variant
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('font', size=6)

    D = df_best_methods.groupby(["dataset_variant"]).best_algorithm.value_counts()
    D = D.reset_index(name="count")
    D.rename(columns={"index": "best_algorithm"}, inplace=True)

    best_methods_count = {}
    for dataset_variant in D.dataset_variant.unique():
        best_methods_count[dataset_variant] = {}
    
    # iterate over rows with iterrows()
    for index, row in D.iterrows():
        alg_list = eval(row["best_algorithm"])
        for alg in alg_list:
            try:
                best_methods_count[row["dataset_variant"]][alg] += row["count"]
            except KeyError:
                best_methods_count[row["dataset_variant"]][alg] = row["count"]

    D = pd.DataFrame.from_dict(best_methods_count, orient='index')
    D = D.reset_index()
    D.rename(columns={"index": "Dataset Variant"}, inplace=True)
    D = D.sort_values(by="Dataset Variant", ascending=False)
    D = D.melt(id_vars=["Dataset Variant"], var_name="Best Algorithm", value_name="Count")
    D.dropna(inplace=True) # Removing algorithms that are not in the base dataset
    # Replace count with proportion
    D["Count"] = D["Count"] / D.groupby(["Dataset Variant"])["Count"].transform('sum')
    D = D.sort_values(by=["Dataset Variant", "Count"], ascending=False)
    D.rename(columns={"Count": "Proportion"}, inplace=True)

    g = sns.catplot(y="Dataset Variant", x="Proportion", hue="Best Algorithm", data=D, kind="bar", errorbar=None, legend=False, height=2, aspect=1.5, palette="husl", hue_order=hue_order)
    plt.legend(loc="best", borderaxespad=0., fontsize=5)
    # xticks set
    g.set(xticks=[0, 0.2, 0.4, 0.6])
    g.set_xticklabels(["0", "0.2", "0.4", "0.6"])
    plt.xlabel("Proportion")
    plt.ylabel("Dataset Variant")
    plt.tight_layout()
    filename = f"plots/{args.n_classes}-best-algorithms-count-per-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Plot the count of the best algorithm per base dataset and dataset variant
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('font', size=6)

    D = df_best_methods.groupby(["base_dataset", "dataset_variant"]).best_algorithm.value_counts()
    D = D.reset_index(name="count")
    best_methods_count = {}
    for base_dataset in D.base_dataset.unique():
        best_methods_count[base_dataset] = {}
        for dataset_variant in D.dataset_variant.unique():
            best_methods_count[base_dataset][dataset_variant] = {}

    D.rename(columns={"index": "best_algorithm"}, inplace=True)

    # iterate over rows with iterrows()
    for index, row in D.iterrows():
        alg_list = eval(row["best_algorithm"])
        for alg in alg_list:
            try:
                best_methods_count[row["base_dataset"]][row["dataset_variant"]][alg] += row["count"]
            except KeyError:
                best_methods_count[row["base_dataset"]][row["dataset_variant"]][alg] = row["count"]
    
    D = pd.DataFrame.from_dict(best_methods_count, orient='index')
    D = D.reset_index()
    D.rename(columns={"index": "Base Dataset"}, inplace=True)
    D = D.sort_values(by="Base Dataset", ascending=False)
    D = D.melt(id_vars=["Base Dataset"], var_name="Dataset Variant", value_name="Count")
    # Convert dictionaries to columns
    D_expanded = pd.concat([D['Count'].apply(pd.Series)], axis=1)
    D = pd.concat([D, D_expanded], axis=1)
    D = D.drop('Count', axis=1)
    D.replace({np.nan: 0}, inplace=True)
    D = D.melt(id_vars=["Base Dataset", "Dataset Variant"], var_name="Best Algorithm", value_name="Count")
    # Replace count with proportion
    D["Count"] = D["Count"] / D.groupby(["Base Dataset", "Dataset Variant"])["Count"].transform('sum')
    D = D.sort_values(by=["Base Dataset", "Dataset Variant", "Count"], ascending=False)
    D.rename(columns={"Count": "Proportion"}, inplace=True)
    
    # Plot (Base Dataset, Dataset Variant) combinations
    g = sns.catplot(y="Base Dataset", x="Proportion", hue="Best Algorithm", col="Dataset Variant", data=D, kind="bar", errorbar=None, legend=False, height=2, aspect=1.5, palette="husl", col_wrap=2, hue_order=hue_order)
    g.set_titles("{col_name}")
    plt.legend(loc="center right", borderaxespad=0., fontsize=5)
    plt.xlabel("Proportion")
    plt.ylabel("Base Dataset")
    plt.tight_layout()
    filename = f"plots/{args.n_classes}-best-algorithms-count-per-base-dataset-and-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

elif args.plot_type == "table-all-results":
    get_performance = lambda x: f"{np.round(x.f1_test.mean(), 4)} ({np.round(1.96 * np.std(x.f1_test.values)/np.sqrt(len(x.f1_test.values)), 4)})"

    # Dataframe with all results
    df_results = pd.DataFrame(columns=["Dataset", "Algorithm", "Full-Bag K-fold", "Split-Bag K-fold", "Split-Bag Shuffle", "Split-Bag Bootstrap"])

    for dataset in final_results.dataset.unique():
        final_result_dataset = deepcopy(final_results[final_results.dataset == dataset])
        for model in final_result_dataset.model.unique():
            data = final_result_dataset[(final_result_dataset.model == model)]
            df_results = pd.concat([df_results, pd.DataFrame({
                "Dataset": dataset,
                "Algorithm": model,
                "Full-Bag K-fold": get_performance(data[data.split_method == "FB\nKF\n"]),
                "Split-Bag K-fold": get_performance(data[data.split_method == "SB\nKF\n"]),
                "Split-Bag Shuffle": get_performance(data[data.split_method == "SB\nSH\n0.5"]),
                "Split-Bag Bootstrap": get_performance(data[data.split_method == "SB\nBS\n0.5"]),
            }, index=[0])], ignore_index=True)

    with pd.option_context("max_colwidth", 10000):
        df_results.to_latex(buf="tables/all-results.tex", index=False, escape=False, longtable=True)

elif args.plot_type == "table-ci-tests":
    df_ci = pd.read_csv("ci-tests/ci-results.csv")
    df_ci.llp_variant.replace({
        "naive": "Naive",
        "simple": "Simple",
        "intermediate": "Intermediate",
        "hard": "Hard",
    }, inplace=True)

    for col in ["b-indep-y", "x-indep-b", "x-indep-y-given-b", "x-indep-b-given-y", "b-indep-y-given-x"]:
        df_ci[col] = df_ci[col].apply(lambda x: f"{x:.2f}")

    df_ci.rename(columns={
        "dataset": "Dataset",
        "llp_variant": "LLP Variant",
        "follow_dgm": "Follow DGM",
        "b-indep-y": "B Indep. Y (p-value)",
        "x-indep-b": "X Indep. B (p-value)",
        "x-indep-y-given-b": "X Indep. Y Given B (p-value)",
        "x-indep-b-given-y": "X Indep. B Given Y (p-value)",
        "b-indep-y-given-x": "B Indep. Y Given X (p-value)",
    }, inplace=True)
    
    with pd.option_context("max_colwidth", 10000):
        df_ci.to_latex(buf="tables/all-ci-tests.tex", index=False, escape=False, longtable=True)
