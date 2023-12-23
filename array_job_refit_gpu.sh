#!/bin/bash -l

# Set SCC project

# Submit an array job with 720 tasks
#$ -t 1-240

# Specify hard time limit for the job.
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=12:00:00

# Give job a name
#$ -N llp-benchmark

#$ -pe omp 8
#$ -l mem_per_core=1G

# Request 1 GPU 
#$ -l gpus=1

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=6.0

#$ -v LD_LIBRARY_PATH=$R_HOME/lib:$LD_LIBRARY_PATH	

declare -a params
idx=0
IFS=' ' # space is set as delimiter

# Binary
# for dataset in "adult-hard-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-25" "adult-hard-extra-large-fol-clust-fol-clust-cluster-kmeans-20" "adult-hard-large-equal-close-global-cluster-kmeans-5" "adult-hard-large-equal-far-global-cluster-kmeans-5" "adult-hard-large-equal-mixed-cluster-kmeans-5" "adult-hard-large-not-equal-close-global-cluster-kmeans-5" "adult-hard-large-not-equal-far-global-cluster-kmeans-5" "adult-hard-large-not-equal-mixed-cluster-kmeans-5" "adult-hard-massive-fol-clust-fol-clust-cluster-kmeans-30" "adult-hard-small-equal-close-global-cluster-kmeans-5" "adult-hard-small-equal-far-global-cluster-kmeans-5" "adult-hard-small-equal-mixed-cluster-kmeans-5" "adult-hard-small-not-equal-close-global-cluster-kmeans-5" "adult-hard-small-not-equal-far-global-cluster-kmeans-5" "adult-hard-small-not-equal-mixed-cluster-kmeans-5" "adult-intermediate-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-25" "adult-intermediate-extra-large-fol-clust-fol-clust-cluster-kmeans-20" "adult-intermediate-large-equal-close-global-cluster-kmeans-5" "adult-intermediate-large-equal-far-global-cluster-kmeans-5" "adult-intermediate-large-equal-mixed-cluster-kmeans-5" "adult-intermediate-large-not-equal-close-global-cluster-kmeans-5" "adult-intermediate-large-not-equal-far-global-cluster-kmeans-5" "adult-intermediate-large-not-equal-mixed-cluster-kmeans-5" "adult-intermediate-massive-fol-clust-fol-clust-cluster-kmeans-30" "adult-intermediate-small-equal-close-global-cluster-kmeans-5" "adult-intermediate-small-equal-far-global-cluster-kmeans-5" "adult-intermediate-small-equal-mixed-cluster-kmeans-5" "adult-intermediate-small-not-equal-close-global-cluster-kmeans-5" "adult-intermediate-small-not-equal-far-global-cluster-kmeans-5" "adult-intermediate-small-not-equal-mixed-cluster-kmeans-5" "adult-naive-extra-extra-large-fol-clust-None-cluster-kmeans-25" "adult-naive-extra-large-fol-clust-None-cluster-kmeans-20" "adult-naive-large-equal-None-cluster-None-None" "adult-naive-large-not-equal-None-cluster-None-None" "adult-naive-massive-fol-clust-None-cluster-kmeans-30" "adult-naive-small-equal-None-cluster-None-None" "adult-naive-small-not-equal-None-cluster-None-None" "adult-simple-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-25" "adult-simple-extra-large-fol-clust-fol-clust-cluster-kmeans-20" "adult-simple-large-equal-close-global-cluster-None-None" "adult-simple-large-equal-far-global-cluster-None-None" "adult-simple-large-equal-mixed-cluster-None-None" "adult-simple-large-not-equal-close-global-cluster-None-None" "adult-simple-large-not-equal-far-global-cluster-None-None" "adult-simple-large-not-equal-mixed-cluster-None-None" "adult-simple-massive-fol-clust-fol-clust-cluster-kmeans-30" "adult-simple-small-equal-close-global-cluster-None-None" "adult-simple-small-equal-far-global-cluster-None-None" "adult-simple-small-equal-mixed-cluster-None-None" "adult-simple-small-not-equal-close-global-cluster-None-None" "adult-simple-small-not-equal-far-global-cluster-None-None" "adult-simple-small-not-equal-mixed-cluster-None-None"
# do
#     for model in "llpfc"
#     do
#         for loss in "abs"
# 		do
#             for n_split in "5"
#             do
#                 for splitter in "split-bag-shuffle"
#                 do
#                     for validation_size_perc in "0.5"
#                     do
#                         for exec in "-1"
#                         do
#                             params[idx]=$dataset$IFS$model$IFS$loss$IFS$splitter$IFS$n_split$IFS$validation_size_perc$IFS$exec
#                             ((idx++))
#                         done
#                     done
#                 done
#             done
# 		done
#     done
# done

# Binary
# for dataset in "cifar-10-grey-animal-vehicle-hard-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40" "cifar-10-grey-animal-vehicle-hard-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30" "cifar-10-grey-animal-vehicle-hard-large-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-large-equal-far-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-large-equal-mixed-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-large-not-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-large-not-equal-far-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-large-not-equal-mixed-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50" "cifar-10-grey-animal-vehicle-hard-small-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-small-equal-far-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-small-equal-mixed-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-small-not-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-small-not-equal-far-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-hard-small-not-equal-mixed-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-intermediate-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40" "cifar-10-grey-animal-vehicle-intermediate-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30" "cifar-10-grey-animal-vehicle-intermediate-large-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-intermediate-large-not-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-intermediate-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50" "cifar-10-grey-animal-vehicle-intermediate-small-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-intermediate-small-not-equal-close-global-cluster-kmeans-5" "cifar-10-grey-animal-vehicle-naive-extra-extra-large-fol-clust-None-cluster-kmeans-autoencoder-40" "cifar-10-grey-animal-vehicle-naive-extra-large-fol-clust-None-cluster-kmeans-autoencoder-30" "cifar-10-grey-animal-vehicle-naive-large-equal-None-cluster-None-None" "cifar-10-grey-animal-vehicle-naive-large-not-equal-None-cluster-None-None" "cifar-10-grey-animal-vehicle-naive-massive-fol-clust-None-cluster-kmeans-autoencoder-50" "cifar-10-grey-animal-vehicle-naive-small-equal-None-cluster-None-None" "cifar-10-grey-animal-vehicle-naive-small-not-equal-None-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40" "cifar-10-grey-animal-vehicle-simple-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30" "cifar-10-grey-animal-vehicle-simple-large-equal-close-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-large-equal-far-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-large-equal-mixed-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-large-not-equal-close-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-large-not-equal-far-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-large-not-equal-mixed-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50" "cifar-10-grey-animal-vehicle-simple-small-equal-close-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-small-equal-far-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-small-equal-mixed-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-small-not-equal-close-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-small-not-equal-far-global-cluster-None-None" "cifar-10-grey-animal-vehicle-simple-small-not-equal-mixed-cluster-None-None"
# do
#     for model in "llpfc"
#     do
#         for loss in "abs"
# 		do
#             for n_split in "5"
#             do
#                 for splitter in "split-bag-shuffle"
#                 do
#                     for validation_size_perc in "0.5"
#                     do
#                         for ((exec=0; exec<5; exec++))
#                         do
#                             params[idx]=$dataset$IFS$model$IFS$loss$IFS$splitter$IFS$n_split$IFS$validation_size_perc$IFS$exec
#                             ((idx++))
#                         done
#                     done
#                 done
#             done
# 		done
#     done
# done


# for dataset in "cifar-10-hard-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40" "cifar-10-hard-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30" "cifar-10-hard-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50" "cifar-10-intermediate-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40" "cifar-10-intermediate-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30" "cifar-10-intermediate-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50" "cifar-10-naive-extra-extra-large-fol-clust-None-cluster-kmeans-autoencoder-40" "cifar-10-naive-extra-large-fol-clust-None-cluster-kmeans-autoencoder-30" "cifar-10-naive-massive-fol-clust-None-cluster-kmeans-autoencoder-50" "cifar-10-simple-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40" "cifar-10-simple-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30" "cifar-10-simple-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50"
# do
#     for model in "mixbag" "llp-vat" "llpfc" "dllp"
#     do
#         for loss in "abs"
# 		do
#             for n_split in "5"
#             do
#                 for splitter in "split-bag-shuffle"
#                 do
#                     for validation_size_perc in "0.5"
#                     do
#                         for ((exec=0; exec<5; exec++))
#                         do
#                             params[idx]=$dataset$IFS$model$IFS$loss$IFS$splitter$IFS$n_split$IFS$validation_size_perc$IFS$exec
#                             ((idx++))
#                         done
#                     done
#                 done
#             done
# 		done
#     done
# done

# for dataset in "cifar-10-simple-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50"
# do
#     for model in "llp-vat"
#     do
#         for loss in "abs"
# 		do
#             for n_split in "5"
#             do
#                 for splitter in "split-bag-shuffle"
#                 do
#                     for validation_size_perc in "0.5"
#                     do
#                         for exec in "4"
#                         do
#                             params[idx]=$dataset$IFS$model$IFS$loss$IFS$splitter$IFS$n_split$IFS$validation_size_perc$IFS$exec
#                             ((idx++))
#                         done
#                     done
#                 done
#             done
# 		done
#     done
# done

# for dataset in "cifar-10-simple-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50"
# do
#     for model in "llpfc"
#     do
#         for loss in "abs"
# 		do
#             for n_split in "5"
#             do
#                 for splitter in "split-bag-shuffle"
#                 do
#                     for validation_size_perc in "0.5"
#                     do
#                         for exec in "0"
#                         do
#                             params[idx]=$dataset$IFS$model$IFS$loss$IFS$splitter$IFS$n_split$IFS$validation_size_perc$IFS$exec
#                             ((idx++))
#                         done
#                     done
#                 done
#             done
# 		done
#     done
# done

for dataset in "svhn-hard-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40" "svhn-hard-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30" "svhn-hard-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50" "svhn-intermediate-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40" "svhn-intermediate-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30" "svhn-intermediate-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50" "svhn-naive-extra-extra-large-fol-clust-None-cluster-kmeans-autoencoder-40" "svhn-naive-extra-large-fol-clust-None-cluster-kmeans-autoencoder-30" "svhn-naive-massive-fol-clust-None-cluster-kmeans-autoencoder-50" "svhn-simple-extra-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-40" "svhn-simple-extra-large-fol-clust-fol-clust-cluster-kmeans-autoencoder-30" "svhn-simple-massive-fol-clust-fol-clust-cluster-kmeans-autoencoder-50"
do
    for model in "mixbag" "llp-vat" "llpfc" "dllp"
    do
        for loss in "abs"
		do
            for n_split in "5"
            do
                for splitter in "split-bag-shuffle"
                do
                    for validation_size_perc in "0.5"
                    do
                        for ((exec=0; exec<5; exec++))
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


index=$(($SGE_TASK_ID-1))
read -ra taskinput <<< "${params[$index]}" # str is read into an array as tokens separated by IFS

cd /usr3/graduate/gvfranco/
source .bashrc
cd /usr3/graduate/gvfranco/kdd24-llp-benchmark/
module load R/4.1.2
module load python3/3.8.16
source /project/mcnet/venv3.8/bin/activate
module load rpy2/3.5.7
module load pytorch/1.13.1
#fix_rpy2.sh
export LD_LIBRARY_PATH=$R_HOME/lib:$LD_LIBRARY_PATH

if [ ${#taskinput[@]} -eq 7 ]
then
    python3 run_experiments.py -d ${taskinput[0]} -m ${taskinput[1]} -l ${taskinput[2]} -s ${taskinput[3]} -n ${taskinput[4]} -v ${taskinput[5]} -e ${taskinput[6]}
elif [ ${#taskinput[@]} -eq 6 ]
then
    python3 run_experiments.py -d ${taskinput[0]} -m ${taskinput[1]} -l ${taskinput[2]} -s ${taskinput[3]} -n ${taskinput[4]} -e ${taskinput[5]}
fi

# index=1
# for i in "${params[@]}"; do # access each element of array
#    echo "$index: $i"
#    index=$((index+1))
# done
