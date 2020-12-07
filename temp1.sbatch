#! /bin/bash
#SBATCH --job-name=glow-pretrain
#SBATCH --error=./sbatch_logs/glow-pretrain-%j.err
#SBATCH --output=./sbatch_logs/glow-pretrain-%j.Log
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=2
#SBATCH --nodes=4
#SBATCH --gpus=4
module load Python/Anaconda_v10.2019
source activate speech2face
srun python train.py

