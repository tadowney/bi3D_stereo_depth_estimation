#!/bin/bash
#
#SBATCH --job-name=leastereo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=leastereo.out
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tdowney@scu.edu
#
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
CUDA_VISIBLE_DEVICES=0 python /WAVE/users/unix/tdowney/DeepLearning/project/LEAStereo-master/LEAStereo-master/train.py --batch_size=4 \
                --crop_height=384 \
                --crop_width=576 \
                --maxdisp=192 \
                --threads=8 \
                --save_path='./run/sceneflow/retrain/' \
                --fea_num_layer 6 --mat_num_layers 12 \
                --fea_filter_multiplier 8 --fea_block_multiplier 4 --fea_step 3  \
                --mat_filter_multiplier 8 --mat_block_multiplier 4 --mat_step 3  \
                --net_arch_fea='run/sceneflow/best/architecture/feature_network_path.npy' \
                --cell_arch_fea='run/sceneflow/best/architecture/feature_genotype.npy' \
                --net_arch_mat='run/sceneflow/best/architecture/matching_network_path.npy' \
                --cell_arch_mat='run/sceneflow/best/architecture/matching_genotype.npy' \
                --nEpochs=20 2>&1 |tee ./run/sceneflow/retrain/log.txt
