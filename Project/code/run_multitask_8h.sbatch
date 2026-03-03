#!/bin/bash
#SBATCH -p defq
#SBATCH --nodelist=node221
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpunode
#SBATCH -t 16:00:00
#SBATCH -J cs224n_multitask
#SBATCH -o logs/%x-%j.out

echo "Job started on $(hostname)"
echo "Time: $(date)"

source ~/.bashrc
conda activate cs224n_dfp

module load cuda12.6/toolkit/12.6
nvidia-smi


python classifier.py --fine-tune-mode last-linear-layer --use_gpu --lr 1e-3
python classifier.py --fine-tune-mode full-model --use_gpu --lr 1e-5

python multitask_classifier.py --fine-tune-mode last-linear-layer --use_gpu --lr 1e-5
python multitask_classifier.py --fine-tune-mode full-model --use_gpu --lr 1e-5

echo "Finished at $(date)"
