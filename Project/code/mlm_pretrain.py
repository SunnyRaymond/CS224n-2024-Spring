import argparse
import csv
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bert import BertModel
from optimizer import AdamW
from tokenizer import BertTokenizer


TQDM_DISABLE = False


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preprocess_string(s):
    return " ".join(
        s.lower()
        .replace(".", " .")
        .replace("?", " ?")
        .replace(",", " ,")
        .replace("'", " '")
        .split()
    )


def load_unlabeled_sentences(sst_path, para_path, sts_path):
    sents = []

    with open(sst_path, "r", encoding="utf-8") as fp:
        for row in csv.DictReader(fp, delimiter="\t"):
            sents.append(row["sentence"].lower().strip())

    with open(para_path, "r", encoding="utf-8") as fp:
        for row in csv.DictReader(fp, delimiter="\t"):
            sents.append(preprocess_string(row["sentence1"]))
            sents.append(preprocess_string(row["sentence2"]))

    with open(sts_path, "r", encoding="utf-8") as fp:
        for row in csv.DictReader(fp, delimiter="\t"):
            sents.append(preprocess_string(row["sentence1"]))
            sents.append(preprocess_string(row["sentence2"]))

    print(f"Loaded {len(sents)} unlabeled sentences for MLM pretraining.")
    return sents


class MLMSentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class MLMCollator:
    def __init__(self, tokenizer, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id

    def _mask_tokens(self, input_ids):
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        special_tokens = (
            (input_ids == self.pad_id)
            | (input_ids == self.cls_id)
            | (input_ids == self.sep_id)
        )
        probability_matrix.masked_fill_(special_tokens, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100

        replace_prob = torch.rand(labels.shape)
        indices_replaced = masked_indices & (replace_prob < 0.8)
        input_ids[indices_replaced] = self.mask_id

        indices_random = masked_indices & (replace_prob >= 0.8) & (replace_prob < 0.9)
        random_words = torch.randint(
            low=0, high=self.tokenizer.vocab_size, size=labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, labels

    def __call__(self, sentences):
        enc = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = torch.LongTensor(enc["input_ids"])
        attention_mask = torch.LongTensor(enc["attention_mask"])
        input_ids, labels = self._mask_tokens(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class BertForMLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden = self.bert.config.hidden_size
        vocab = self.bert.config.vocab_size

        self.mlm_dense = nn.Linear(hidden, hidden)
        self.mlm_layer_norm = nn.LayerNorm(hidden, eps=self.bert.config.layer_norm_eps)
        self.mlm_decoder = nn.Linear(hidden, vocab, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab))
        self.mlm_decoder.weight = self.bert.word_embedding.weight

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = out["last_hidden_state"]
        x = self.mlm_dense(x)
        x = F.gelu(x)
        x = self.mlm_layer_norm(x)
        x = self.mlm_decoder(x) + self.mlm_bias
        return x


def save_checkpoint(model, optimizer, args, filepath):
    payload = {
        "bert_state_dict": model.bert.state_dict(),
        "model_state_dict": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }
    torch.save(payload, filepath)
    print(f"Saved MLM checkpoint to {filepath}")


def train_mlm(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    sents = load_unlabeled_sentences(
        sst_path=args.sst_train,
        para_path=args.para_train,
        sts_path=args.sts_train,
    )
    ds = MLMSentenceDataset(sents)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    collator = MLMCollator(tokenizer, mlm_prob=args.mlm_prob)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    model = BertForMLM().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dl, desc=f"mlm-train-{epoch}", disable=TQDM_DISABLE):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch}: mlm loss :: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, args, args.output)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--output", type=str, default="domain-mlm-pretrain.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    train_mlm(args)
