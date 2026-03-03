# CS 224N Default Final Project - Multitask BERT

This is the default final project for the Stanford CS 224N class. Please refer to the project handout on the course website for detailed instructions and an overview of the codebase.

This project comprises two parts. In the first part, you will implement some important components of the BERT model to better understand its architecture.
In the second part, you will use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection, and semantic similarity. You will implement extensions to improve your model's performance on the three downstream tasks.

In broad strokes, Part 1 of this project targets:

- bert.py: Missing code blocks.
- classifier.py: Missing code blocks.
- optimizer.py: Missing code blocks.

And Part 2 targets:

- multitask_classifier.py: Missing code blocks.
- datasets.py: Possibly useful functions/classes for extensions.
- evaluation.py: Possibly useful functions/classes for extensions.

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Additional Pretraining (Domain-Adaptive MLM)

Run from the `Project/code` directory.

1. MLM pretraining on target-domain training text:

```bash
python mlm_pretrain.py --use_gpu --epochs 2 --batch_size 16 --lr 5e-5 --output domain-mlm-pretrain.pt
```

2. Multitask training initialized from the MLM-pretrained BERT:

```bash
python multitask_classifier.py --fine-tune-mode last-linear-layer --use_gpu --lr 1e-5 --pretrained_bert_path domain-mlm-pretrain.pt
python multitask_classifier.py --fine-tune-mode full-model --use_gpu --lr 1e-5 --pretrained_bert_path domain-mlm-pretrain.pt
```

3. Optional single-command SLURM run:

```bash
sbatch run_multitask_8h.sbatch
```

## Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
