#!/bin/bash
#SBATCH -p defq
#SBATCH --nodelist=node221
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpunode
#SBATCH -t 7-00:00:00
#SBATCH -J cs224n
#SBATCH -o logs/%x-%j.out

echo "Job started on $(hostname)"
echo "Time: $(date)"

source ~/.bashrc
conda activate cs224n_dfp

module load cuda12.6/toolkit/12.6
nvidia-smi

python mlm_pretrain.py --use_gpu --epochs 2 --batch_size 16 --lr 5e-5 --output domain-mlm-pretrain.pt

python multitask_classifier.py --fine-tune-mode last-linear-layer --use_gpu --lr 1e-5 --pretrained_bert_path domain-mlm-pretrain.pt
python multitask_classifier.py --fine-tune-mode full-model --use_gpu --lr 1e-5 --pretrained_bert_path domain-mlm-pretrain.pt

echo "Finished at $(date)"
