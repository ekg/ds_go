#!/bin/bash
#SBATCH -A BIF148                 # embedding life project
#SBATCH -J autotp_test            # Job name
#SBATCH -o %x-%j.out              # Output file name (%x=job name, %j=job id)
#SBATCH -e %x-%j.err              # Error file name
#SBATCH -t 01:00:00               # Maximum job time (HH:MM:SS)
#SBATCH -p batch                  # batch queue
#SBATCH -q debug                  # debugging QOS
#SBATCH --nodes=4                      # 4 Frontier nodes
#SBATCH --ntasks-per-node=1            # 1 launcher per node
#SBATCH --gpus-per-node=8              # 8 GPUs (MI250X) per node
#SBATCH --cpus-per-task=7              # CPU cores for data loading
#SBATCH --time=01:00:00
#SBATCH --exclusive               # Request exclusive access to node

module load rocm

export OMP_NUM_THREADS=7

# Master for NCCL
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

# Build hostfile for DeepSpeed launcher
scontrol show hostnames $SLURM_JOB_NODELIST | awk '{print $1" slots=8"}' > hostfile

# Set up micromamba environment
export MAMBA_EXE='/autofs/nccs-svm1_home1/erikgarrison/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/lustre/orion/scratch/erikgarrison/bif148/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
micromamba activate gruboros

# Launch with DeepSpeed CLI (auto sets up TP size=8 & DP across 4 nodes)
srun --gpu-bind=closest deepspeed \
  --hostfile hostfile \
  --num_nodes 4 \
  --num_gpus 8 \
  train.py
