#!/bin/bash
#SBATCH --job-name=autotp_test
#SBATCH --nodes=4                      # 4 Frontier nodes
#SBATCH --ntasks-per-node=1            # 1 launcher per node
#SBATCH --gpus-per-node=8              # 8 GPUs (MI250X) per node
#SBATCH --cpus-per-task=7              # CPU cores for data loading
#SBATCH --time=01:00:00
#SBATCH --partition=standard

module load rocm
module load deepspeed/0.16.8

export OMP_NUM_THREADS=7

# Master for NCCL
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

# Build hostfile for DeepSpeed launcher
scontrol show hostnames $SLURM_JOB_NODELIST | awk '{print $1" slots=8"}' > hostfile

# Launch with DeepSpeed CLI (auto sets up TP size=8 & DP across 4 nodes)
srun --gpu-bind=closest deepspeed \
  --hostfile hostfile \
  --num_nodes 4 \
  --num_gpus 8 \
  train.py
