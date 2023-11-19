import glob

# files = glob.glob("datasets-ci/*massive*")
# files += glob.glob("datasets-ci/*extra-large*")
# files = sorted(files)

# for file in files:
#     dataset = file.split("/")[1].split(".")[0]
#     print(f'"{dataset}"', end=" ")

datasets = ["adult-hard-small-equal-close-global-cluster-kmeans-5","adult-hard-small-equal-far-global-cluster-kmeans-5","adult-hard-small-equal-mixed-cluster-kmeans-5","adult-hard-small-not-equal-close-global-cluster-kmeans-5","adult-hard-small-not-equal-far-global-cluster-kmeans-5","adult-hard-small-not-equal-mixed-cluster-kmeans-5","adult-intermediate-small-equal-close-global-cluster-kmeans-5","adult-intermediate-small-equal-far-global-cluster-kmeans-5","adult-intermediate-small-equal-mixed-cluster-kmeans-5","adult-intermediate-small-not-equal-close-global-cluster-kmeans-5","adult-intermediate-small-not-equal-far-global-cluster-kmeans-5","adult-intermediate-small-not-equal-mixed-cluster-kmeans-5","adult-naive-small-equal-None-cluster-None-None","adult-naive-small-not-equal-None-cluster-None-None","adult-simple-small-equal-close-global-cluster-None-None","adult-simple-small-equal-far-global-cluster-None-None","adult-simple-small-equal-mixed-cluster-None-None","adult-simple-small-not-equal-close-global-cluster-None-None","adult-simple-small-not-equal-far-global-cluster-None-None","adult-simple-small-not-equal-mixed-cluster-None-None","cifar-10-grey-animal-vehicle-hard-small-equal-close-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-hard-small-equal-far-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-hard-small-equal-mixed-cluster-kmeans-5","cifar-10-grey-animal-vehicle-hard-small-not-equal-close-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-hard-small-not-equal-far-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-hard-small-not-equal-mixed-cluster-kmeans-5","cifar-10-grey-animal-vehicle-intermediate-small-equal-close-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-intermediate-small-not-equal-close-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-naive-small-equal-None-cluster-None-None","cifar-10-grey-animal-vehicle-naive-small-not-equal-None-cluster-None-None","cifar-10-grey-animal-vehicle-simple-small-equal-close-global-cluster-None-None","cifar-10-grey-animal-vehicle-simple-small-equal-far-global-cluster-None-None","cifar-10-grey-animal-vehicle-simple-small-equal-mixed-cluster-None-None","cifar-10-grey-animal-vehicle-simple-small-not-equal-close-global-cluster-None-None","cifar-10-grey-animal-vehicle-simple-small-not-equal-far-global-cluster-None-None","cifar-10-grey-animal-vehicle-simple-small-not-equal-mixed-cluster-None-None"]

files = []
for dataset in datasets:
    files += glob.glob(f"datasets-experiments-results/{dataset}*")

files = sorted([f for f in files if "shuffle" in f])

new_files = []

for file in files:
    execution = int(file.split("_")[-1].split(".")[0])
    if execution <= 4:
        new_files.append(file)

files = new_files

import shutil


for file in files:
    # 2nd option
    shutil.copy(file, "files-copy/")

