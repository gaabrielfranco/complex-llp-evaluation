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
parser.add_argument('--n_classes', "-n", choices=["binary", "multiclass", "all"], type=str, required=True, help="Number of classes")
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
    "mixbag": "MixBag",
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

if args.n_classes == "all":
    pass
elif args.n_classes == "binary":
    final_results = final_results[((final_results.base_dataset != "cifar-10") & (final_results.base_dataset != "svhn"))]
else:
    final_results = final_results[((final_results.base_dataset == "cifar-10") | (final_results.base_dataset == "svhn"))]

# Getting all models
all_models = final_results["model"].unique()

# Adding domain column
final_results["dataset_domain"] = final_results["bag_sizes"].apply(lambda x: "large" if x == "fol-clust" else "small")

if args.plot_type == "check-n-experiments":
    total_models = len(final_results)
    print("Total trained models: ", total_models)
    for model in final_results["model"].unique():
        print(model, len(final_results[final_results["model"] == model]))
    print("")
    T = {
        "MM": 4,
        "AMM": 12,
        "LMM": 36,
        "EM/LR": 6,
        "DLLP": 5,
        "LLP-VAT": 5,
        "MixBag": 5,
        "LLPFC": 5,
    }

    print("Total trained models considering HS: ")
    sum_models = 0
    for model in final_results["model"].unique():
        print(model, len(final_results[final_results["model"] == model]) * (T[model] + 1))
        sum_models += len(final_results[final_results["model"] == model]) * (T[model] + 1)
    print("Total models: ", sum_models)
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

    # Removing base dataset for multiclass
    files = [file for file in files if "cifar-10.parquet" not in file and "svhn.parquet" not in file]

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
        hue_order = ["DLLP", "LLP-VAT", "MixBag", "LLPFC", "MM", "AMM", "LMM", "EM/LR"]
    else:
        hue_order = ["DLLP", "LLP-VAT", "MixBag", "LLPFC"]

    palette = sns.color_palette()
    palette = palette[:len(hue_order)]

    # Base dataset map
    base_dataset_map = {
        "cifar-10": "CIFAR-10",
        "svhn": "SVHN",
        "cifar-10-grey": "CIFAR-10\n(Grayscale)",
        "adult": "Adult"
    }

    final_results["base_dataset"] = final_results["base_dataset"].replace(base_dataset_map)

    # F1-score
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rc('font', size=6)
    g = sns.catplot(y="base_dataset", x="f1_test", hue="model", col="dataset_variant", 
                    data=final_results, kind="bar", col_order=["Hard", "Intermediate", "Simple", "Naive"], 
                    legend=True, height=1, aspect=1.25, sharex=True, errorbar="ci", capsize=0.05, col_wrap=2, 
                    errwidth=0.9, hue_order=hue_order, palette=palette, legend_out=True)
    g.set_titles("{col_name}")
    # Draw a line with the f-score of the supervised neural network
    for ax in g.axes.flat:
        if args.n_classes == "binary":
            ax.axvline(x=base_datasets_supervised_f1_score["adult"], ymin=0.5, ymax=1, color="black", linestyle="--", label="Adult\nsupervised") # Adult performance
            ax.axvline(x=base_datasets_supervised_f1_score["cifar-10-grey"], ymin=0, ymax=0.5, color="red", linestyle="--", label="CIFAR-10\n(Grayscale)\nsupervised") # CIFAR-10-Grey performance
        else:
            ax.axvline(x=base_datasets_supervised_f1_score["cifar-10"], ymin=0.5, ymax=1, color="black", linestyle="--", label="CIFAR-10\nsupervised") # CIFAR-10 performance
            ax.axvline(x=base_datasets_supervised_f1_score["svhn"], ymin=0, ymax=0.5, color="red", linestyle="--", label="SVHN\nsupervised") # SVHN performance
    
    handles, labels = ax.get_legend_handles_labels()
    # Adding the new handles and labels to the legend
    g._legend.remove()
    # Legend outside the plot
    g.fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.8, 0.5), fontsize=5, ncols=1)
    #g.fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.7, 0.5), fontsize=5, ncol=1)

    # Set xticks
    for idx, ax in enumerate(g.axes.flat):
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_ylabel("")
        if idx == 3:
            ax.set_xticklabels(["0", ".2", ".4", ".6", ".8", "1"])
            ax.set_xlabel(r"$F_1$-score")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    g.set_xlabels(r"$F_1$-score")
    filename = f"plots/{args.n_classes}-avg-performance-per-method-f1.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Heatmap of counts of best f1-score per exec
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('font', size=6)

    df_count = pd.DataFrame(columns=["dataset"] + hue_order)
    # For binary, there is no usage of pretty_dataset_name
    if args.n_classes == "binary":
        final_results["pretty_dataset_name"] = final_results["dataset"]
    else:
        final_results["pretty_dataset_name"] = final_results["base_dataset"] + " (" + final_results["dataset_variant"] + ", " + final_results["dataset"].apply(lambda x: x.split("-")[-1]) + " bags)"
    for idx, dataset in enumerate(sorted(final_results.pretty_dataset_name.unique())):
        count_models = {}
        for model in final_results.model.unique():
            count_models[model] = 0.0

        best_method = deepcopy(final_results[(final_results.pretty_dataset_name == dataset)])

        if len(best_method) != 240 and len(best_method) != 120:
            raise ValueError("Number of experiments is not 240/120")
                
        for exec in sorted(best_method.exec.unique()):
            best_method_exec = deepcopy(best_method[best_method.exec == exec])
            if len(best_method_exec) != 8 and len(best_method_exec) != 4:
                raise ValueError("Number of experiments is not 8/4")
       
            # Model with largest f1-score
            model = best_method_exec[best_method_exec.f1_test == best_method_exec.f1_test.max()].model.values[0]

            count_models[model] += 1
            
        if args.n_classes == "binary":
            df_count = pd.concat([df_count, pd.DataFrame({
                "dataset": dataset,
                "DLLP": count_models["DLLP"],
                "LLP-VAT": count_models["LLP-VAT"],
                "MixBag": count_models["MixBag"],
                "LLPFC": count_models["LLPFC"],
                "MM": count_models["MM"],
                "AMM": count_models["AMM"],
                "LMM": count_models["LMM"],
                "EM/LR": count_models["EM/LR"]
            }, index=[0])], ignore_index=True)
        elif args.n_classes == "multiclass":
            df_count = pd.concat([df_count, pd.DataFrame({
                "dataset": dataset,
                "DLLP": count_models["DLLP"],
                "LLP-VAT": count_models["LLP-VAT"],
                "MixBag": count_models["MixBag"],
                "LLPFC": count_models["LLPFC"]
            }, index=[0])], ignore_index=True)


    # Plot heatmap
    if args.n_classes == "binary":
        fig, ax = plt.subplots(figsize=(4.5, 3))
    else:
        fig, ax = plt.subplots(figsize=(2.75, 3))
    g = sns.heatmap(df_count.set_index("dataset"), annot=False, cmap="YlGnBu", ax=ax)
    # Get axis
    if args.n_classes == "binary":
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.hlines([15, 30, 37, 52, 67, 74, 81], *ax.get_xlim(), color="red")
        # Add text to the heatmap (left side) to substitute the y-axis
        ax.text(-0.5, 7.5, "Adult\n(Hard)", ha="center", va="center", fontsize=6)
        ax.text(-0.5, 22.5, "Adult\n(Interm.)", ha="center", va="center", fontsize=6)
        ax.text(-0.5, 33.5, "Adult\n(Naive)", ha="center", va="center", fontsize=6)
        ax.text(-0.5, 44.5, "Adult\n(Simple)", ha="center", va="center", fontsize=6)
        
        ax.text(-0.5, 59.5, "CIFAR-10\n(Hard)", ha="center", va="center", fontsize=6)
        ax.text(-0.5, 70.5, "CIFAR-10\n(Interm.)", ha="center", va="center", fontsize=6)
        ax.text(-0.5, 78.5, "CIFAR-10\n(Naive)", ha="center", va="center", fontsize=6)
        ax.text(-0.5, 88, "CIFAR-10\n(Simple)", ha="center", va="center", fontsize=6)
    else:
        ax.hlines([3, 6, 9, 12, 15, 18, 21], *ax.get_xlim(), color="red")
        ax.set_ylabel("")
    
    
    plt.savefig(f"plots/heatmap-best-methods-{args.n_classes}.pdf", bbox_inches='tight', pad_inches=0.01, dpi=800)
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

    # # Computing dataset domain
    df_best_methods["dataset_domain"] = df_best_methods["bag_sizes"].apply(lambda x: "large" if x == "fol-clust" else "small")

    # # Heatmap using Generalized Jaccard Index

    # Computing the Jaccard Index (multiset version)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rc('font', size=6)
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
    
    fig = plt.figure(figsize=(2, 1.5))
    gs0 = matplotlib.gridspec.GridSpec(1,2, width_ratios=[20,3], hspace=0.0)
    gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gs0[1], hspace=0, wspace=6)

    ax = fig.add_subplot(gs0[0])
    cax1 = fig.add_subplot(gs00[0])
    cax2 = fig.add_subplot(gs00[1])
    
    vmin, vmax = 0, 1
    from matplotlib.colors import ListedColormap
    for i, base_dataset in enumerate(df_best_methods.base_dataset.unique()):
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
    sns.heatmap(np.ones((4, 4), dtype=int), mask=~np.eye(4, dtype=bool), cmap=ListedColormap(['white']), annot=False, annot_kws={"size": 5}, cbar=False, ax=ax, xticklabels=["Naive", "Simple", "Intermediate", "Hard"], yticklabels=["Naive", "Simple", "Intermediate", "Hard"])
    #plt.tight_layout()
    filename = f"plots/{args.n_classes}-jaccard-index-heatmap-best-algorithm.pdf"
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
        ax[idx // 2, idx % 2].set_xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
        ax[idx // 2, idx % 2].set_xticklabels(["0.00", "0.05", "0.10", "0.15", "0.20", "0.25"])
    plt.tight_layout()
    filename = f"plots/{args.n_classes}-effect-sizes-dist.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

    # Plot the count of the best algorithm per base dataset and dataset variant
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')
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
    D.rename(columns={"Count": "Fraction"}, inplace=True)

    # Plot (Base Dataset, Dataset Variant) combinations
    g = sns.catplot(y="Base Dataset", x="Fraction", hue="Best Algorithm", col="Dataset Variant", 
                    data=D, kind="bar", errorbar=None, legend=False, height=1.5, aspect=0.9, 
                    palette=palette, col_wrap=4, hue_order=hue_order)
    # Remove ylabels
    for ax in g.axes.flat:
        ax.set_ylabel("")

    g.set_titles("{col_name}")
    plt.legend(loc="center right", borderaxespad=0., fontsize=5)
    plt.xlabel("Fraction")
    filename = f"plots/{args.n_classes}-best-algorithms-count-per-base-dataset-and-dataset-variant.pdf"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01, dpi=800)
    plt.close()

elif args.plot_type == "table-all-results":
    get_performance = lambda x: f"{np.round(x.f1_test.mean(), 4)} ({np.round(1.96 * np.std(x.f1_test.values)/np.sqrt(len(x.f1_test.values)), 4)})"

    # Dataframe with all results
    df_results = pd.DataFrame(columns=["Dataset", "Algorithm", "Average (std) $F_1$-score on test set"])

    for dataset in final_results.dataset.unique():
        final_result_dataset = deepcopy(final_results[final_results.dataset == dataset])
        for model in final_result_dataset.model.unique():
            data = final_result_dataset[(final_result_dataset.model == model)]
            df_results = pd.concat([df_results, pd.DataFrame({
                "Dataset": dataset,
                "Algorithm": model,
                "Average (std) $F_1$-score on test set": get_performance(data[data.split_method == "SB\nSH\n0.5"]),
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
