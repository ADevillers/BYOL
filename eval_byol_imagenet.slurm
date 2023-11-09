#!/bin/bash
#SBATCH -A xwh@v100             # Change to match the right project
#SBATCH --qos=qos_gpu-t3
#SBATCH -C v100-32g
#SBATCH --job-name=expe
#SBATCH --time=20:00:00
#SBATCH --nodes=8               # TODO: Define the number of nodes
#SBATCH --ntasks-per-node=4     # = --gres=gpu:
#SBATCH --gres=gpu:4            # = --ntasks-per-node
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --output=./logs/expe_%j.out
#SBATCH --error=./logs/expe_%j.err

# Move to the path where the script is submitted
cd ${SLURM_SUBMIT_DIR}

# Clean modules
module purge
module unload cuda

# Load used modules
module load cuda/11.2
module load pytorch-gpu/py3/1.10.0

# Echo commands
set -x

# Execute python code
srun python src/eval.py \
    --computer=jeanzay \
    --hardware=multi-gpu \
    --precision=mixed \
    --nb_workers=10 \
    --dataset_name=imagenet \
    --resnet_type=resnet50 \
    --z_dim=256 \
    --nb_epochs_clsf=90 \
    --batch_size_clsf=256 \
    --lr_init_clsf=0.2 \
    --momentum_clsf=0.9 \
    --weight_decay_clsf=0.0 \
    --checkpoint=./checkpoints/weights_byol_imagenet.pt \
