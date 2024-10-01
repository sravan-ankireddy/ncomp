#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --job-name=train_taco_0.0004_h100_bs8_afa_ect
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -t 48:00:00
#SBATCH --output=tacc/out/%x_%j.out
#SBATCH --error=tacc/err/%x_%j.err
#SBATCH --account=NCR23002
#SBATCH --mail-type=all
#SBATCH --mail-user=sravan.ankireddy@utexas.edu

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export NCCL_NET_GDR_LEVEL="SYS"
export NCCL_NET_GDR_READ=1

export HF_HOME=$WORK/hf_cache

CONDA_BASE=$WORK/anaconda3

export PATH="$CONDA_BASE/bin:$PATH"
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate taco

srun bash run.sh