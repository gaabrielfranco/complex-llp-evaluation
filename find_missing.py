import os

directory="llp-benchmark-results/"
#datasets = ["adult-hard-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-25","adult-hard-extra-large-fol-clust-fol-clust-cluster-kmeans-20","adult-hard-massive-fol-clust-fol-clust-cluster-kmeans-30","adult-intermediate-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-25","adult-intermediate-extra-large-fol-clust-fol-clust-cluster-kmeans-20","adult-intermediate-massive-fol-clust-fol-clust-cluster-kmeans-30","adult-naive-extra-extra-large-fol-clust-None-cluster-kmeans-25","adult-naive-extra-large-fol-clust-None-cluster-kmeans-20","adult-naive-massive-fol-clust-None-cluster-kmeans-30","adult-simple-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-25","adult-simple-extra-large-fol-clust-fol-clust-cluster-kmeans-20","adult-simple-massive-fol-clust-fol-clust-cluster-kmeans-30","cifar-10-grey-animal-vehicle-hard-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40","cifar-10-grey-animal-vehicle-hard-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30","cifar-10-grey-animal-vehicle-hard-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50","cifar-10-grey-animal-vehicle-intermediate-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40","cifar-10-grey-animal-vehicle-intermediate-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30","cifar-10-grey-animal-vehicle-intermediate-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50","cifar-10-grey-animal-vehicle-naive-extra-extra-large-fol-clust-None-cluster-kmeans-autoencoder-40","cifar-10-grey-animal-vehicle-naive-extra-large-fol-clust-None-cluster-kmeans-autoencoder-30","cifar-10-grey-animal-vehicle-naive-massive-fol-clust-None-cluster-kmeans-autoencoder-50","cifar-10-grey-animal-vehicle-simple-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40","cifar-10-grey-animal-vehicle-simple-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30","cifar-10-grey-animal-vehicle-simple-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50","adult-hard-large-equal-close-global-cluster-kmeans-5","adult-hard-large-equal-far-global-cluster-kmeans-5","adult-hard-large-equal-mixed-cluster-kmeans-5","adult-hard-large-not-equal-close-global-cluster-kmeans-5","adult-hard-large-not-equal-far-global-cluster-kmeans-5","adult-hard-large-not-equal-mixed-cluster-kmeans-5","adult-intermediate-large-equal-close-global-cluster-kmeans-5","adult-intermediate-large-equal-far-global-cluster-kmeans-5","adult-intermediate-large-equal-mixed-cluster-kmeans-5","adult-intermediate-large-not-equal-close-global-cluster-kmeans-5","adult-intermediate-large-not-equal-far-global-cluster-kmeans-5","adult-intermediate-large-not-equal-mixed-cluster-kmeans-5","adult-naive-large-equal-None-cluster-None-None","adult-naive-large-not-equal-None-cluster-None-None","adult-simple-large-equal-close-global-cluster-None-None","adult-simple-large-equal-far-global-cluster-None-None","adult-simple-large-equal-mixed-cluster-None-None","adult-simple-large-not-equal-close-global-cluster-None-None","adult-simple-large-not-equal-far-global-cluster-None-None","adult-simple-large-not-equal-mixed-cluster-None-None","cifar-10-grey-animal-vehicle-hard-large-equal-close-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-hard-large-equal-far-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-hard-large-equal-mixed-cluster-kmeans-5","cifar-10-grey-animal-vehicle-hard-large-not-equal-close-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-hard-large-not-equal-far-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-hard-large-not-equal-mixed-cluster-kmeans-5","cifar-10-grey-animal-vehicle-intermediate-large-equal-close-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-intermediate-large-not-equal-close-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-naive-large-equal-None-cluster-None-None","cifar-10-grey-animal-vehicle-naive-large-not-equal-None-cluster-None-None","cifar-10-grey-animal-vehicle-simple-large-equal-close-global-cluster-None-None","cifar-10-grey-animal-vehicle-simple-large-equal-far-global-cluster-None-None","cifar-10-grey-animal-vehicle-simple-large-equal-mixed-cluster-None-None","cifar-10-grey-animal-vehicle-simple-large-not-equal-close-global-cluster-None-None","cifar-10-grey-animal-vehicle-simple-large-not-equal-far-global-cluster-None-None","cifar-10-grey-animal-vehicle-simple-large-not-equal-mixed-cluster-None-None","cifar-10-grey-animal-vehicle-simple-small-equal-close-global-cluster-None-None","cifar-10-grey-animal-vehicle-simple-small-equal-mixed-cluster-None-None","adult-intermediate-small-equal-mixed-cluster-kmeans-5","cifar-10-grey-animal-vehicle-simple-small-not-equal-far-global-cluster-None-None","cifar-10-grey-animal-vehicle-hard-small-not-equal-far-global-cluster-kmeans-5","adult-hard-small-not-equal-far-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-simple-small-not-equal-close-global-cluster-None-None","adult-hard-small-equal-far-global-cluster-kmeans-5","adult-hard-small-not-equal-mixed-cluster-kmeans-5","cifar-10-grey-animal-vehicle-hard-small-equal-close-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-naive-small-equal-None-cluster-None-None","cifar-10-grey-animal-vehicle-intermediate-small-equal-close-global-cluster-kmeans-5","adult-intermediate-small-not-equal-close-global-cluster-kmeans-5","adult-simple-small-equal-mixed-cluster-None-None","adult-naive-small-not-equal-None-cluster-None-None","adult-intermediate-small-not-equal-mixed-cluster-kmeans-5","cifar-10-grey-animal-vehicle-intermediate-small-not-equal-close-global-cluster-kmeans-5","adult-naive-small-equal-None-cluster-None-None","adult-hard-small-not-equal-close-global-cluster-kmeans-5","adult-intermediate-small-not-equal-far-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-simple-small-equal-far-global-cluster-None-None","adult-simple-small-not-equal-far-global-cluster-None-None","cifar-10-grey-animal-vehicle-hard-small-not-equal-mixed-cluster-kmeans-5","adult-simple-small-equal-close-global-cluster-None-None","cifar-10-grey-animal-vehicle-hard-small-not-equal-close-global-cluster-kmeans-5","adult-simple-small-not-equal-close-global-cluster-None-None","adult-hard-small-equal-close-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-hard-small-equal-far-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-simple-small-not-equal-mixed-cluster-None-None","adult-simple-small-equal-far-global-cluster-None-None","adult-hard-small-equal-mixed-cluster-kmeans-5","adult-intermediate-small-equal-close-global-cluster-kmeans-5","adult-simple-small-not-equal-mixed-cluster-None-None","adult-intermediate-small-equal-far-global-cluster-kmeans-5","cifar-10-grey-animal-vehicle-naive-small-not-equal-None-cluster-None-None","cifar-10-grey-animal-vehicle-hard-small-equal-mixed-cluster-kmeans-5"]
datasets = ["cifar-10-hard-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40","cifar-10-hard-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30","cifar-10-hard-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50","cifar-10-intermediate-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40","cifar-10-intermediate-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30","cifar-10-intermediate-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50","cifar-10-naive-extra-extra-large-fol-clust-None-cluster-kmeans-autoencoder-40","cifar-10-naive-extra-large-fol-clust-None-cluster-kmeans-autoencoder-30","cifar-10-naive-massive-fol-clust-None-cluster-kmeans-autoencoder-50","cifar-10-simple-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40","cifar-10-simple-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30","cifar-10-simple-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50","svhn-hard-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40","svhn-hard-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30","svhn-hard-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50","svhn-intermediate-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40","svhn-intermediate-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30","svhn-intermediate-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50","svhn-naive-extra-extra-large-fol-clust-None-cluster-kmeans-autoencoder-40","svhn-naive-extra-large-fol-clust-None-cluster-kmeans-autoencoder-30","svhn-naive-massive-fol-clust-None-cluster-kmeans-autoencoder-50","svhn-simple-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40","svhn-simple-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30","svhn-simple-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50"]
models = ["dllp","mixbag","llp-vat","llpfc"]
loss="abs"
n_splits="5"
splitter="split-bag-shuffle"
validation_size_perc="0.5"
execs=range(30)

new_file = open("missing.sh", "w")
count = 0
for dataset in datasets:
    for model in models:
        for exec in execs:
            filename = directory + dataset + "_" + model + "_" + loss + "_" + str(None) + "_" + splitter + "_" + n_splits + "_" + validation_size_perc + "_" + str(exec) + ".parquet"
            if not os.path.exists(filename):
                print(f"Missing file: {filename}")
                count += 1
                # Write it in the .sh file
                # Format
                # for dataset in dataset_name
                # do
                #     for model in model_name
                #     do
                #         for loss in "abs"
                #         do
                #             for n_split in "5"
                #             do
                #                 for splitter in "split-bag-shuffle"
                #                 do
                #                     for validation_size_perc in "0.5"
                #                     do
                #                         for exec in exec_name
                #                         do
                #                             params[idx]=$dataset$IFS$model$IFS$loss$IFS$splitter$IFS$n_split$IFS$validation_size_perc$IFS$exec
                #                             ((idx++))
                #                         done
                #                     done
                #                 done
                #             done
                #         done
                #     done
                # done

                new_file.write(f"""
for dataset in "{dataset}"
do
    for model in "{model}"
    do
        for loss in "abs"
        do
            for n_split in "5"
            do
                for splitter in "split-bag-shuffle"
                do
                    for validation_size_perc in "0.5"
                    do
                        for exec in "{exec}"
                        do
                            params[idx]=$dataset$IFS$model$IFS$loss$IFS$splitter$IFS$n_split$IFS$validation_size_perc$IFS$exec
                            ((idx++))
                        done
                    done
                done
            done
        done
    done
done
""")
                
new_file.close()

print(f"Total missing files: {count}")
