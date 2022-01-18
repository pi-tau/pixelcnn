"""Run the training script to train a model to translate from source language to english.

Example:
    python3 run.py  ""  es  ""  --vocab_size 5000 \
                                --word_embed_size 25 \
                                --char_embed_size 15 \
                                --hidden_size 20 \
                                --batch_size 10 \
                                --max_epochs 1 \
                                --log_file train_log.txt
"""

import argparse
import os
import time
import sys

from src.nmt_model import NMT
from src.train import train
from src.utils import compute_corpus_level_bleu_score, fix_random_seeds, get_data
from src.vocab import Vocab


parser = argparse.ArgumentParser()
parser.add_argument("root", help="Path to data folder.")
parser.add_argument("language",
            help="Source language to be translated. Choose one from ('es', 'de').")
parser.add_argument("destination", help="Path to folder where the model will be saved.")
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--vocab_size", default=50000, type=int)
parser.add_argument("--freq_cutoff", default=2, type=int)
parser.add_argument("--word_embed_size", default=256, type=int)
parser.add_argument("--char_embed_size", default=50, type=int)
parser.add_argument("--hidden_size", default=256, type=int)
parser.add_argument("--dropout_rate", default=0.3, type=float)
parser.add_argument("--kernel_size", default=5, type=int)
parser.add_argument("--padding", default=1, type=int)
parser.add_argument("--learning_rate", default=0.003, type=float)
parser.add_argument("--lr_decay", default=0.5, type=float)
parser.add_argument("--clip_grad", default=5.0, type=float)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--max_epochs", default=50, type=int)
parser.add_argument("--max_num_trial", default=6, type=int)
parser.add_argument("--patience_limit", default=5, type=int)
parser.add_argument("--log_file", default="", type=str)
args = parser.parse_args()

if args.language not in ("es", "de"):
    raise ValueError("Language %s not supported" % args.language)


# Create file to log output during training.
if args.log_file == "":   # print to screen
    stdout = sys.stdout
else:
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    stdout = open(args.log_file, "w")


print("------------------------------------------------------------", file=stdout)
print(f"Training a model to translate from {args.language} to en.", file=stdout)
print("------------------------------------------------------------\n", file=stdout)

print(f"""
##############################
Training parameters:
    Seed:                     {args.seed}
    Vocab size:               {args.vocab_size}
    Frequency cutoff:         {args.freq_cutoff}
    Word embed size:          {args.word_embed_size}
    Char embed size:          {args.char_embed_size}
    Hidden size:              {args.hidden_size}
    Dropout rate:             {args.dropout_rate}
    Kernel size:              {args.kernel_size}
    Padding:                  {args.padding}
    Learning rate:            {args.learning_rate}
    Learning rate decay:      {args.lr_decay}
    Clip grad:                {args.clip_grad}
    Batch size:               {args.batch_size}
    Max Epochs:               {args.max_epochs}
    Max num trial:            {args.max_num_trial}
    Patience limit:           {args.patience_limit}
##############################\n""", file=stdout)

# Fix the seeds for random number generators.
if args.seed is not None: fix_random_seeds(args.seed)


# Read the data.
data_path = args.root + "datasets/%s_en_data/" % args.language
print(f"Reading training data from {data_path} ...", file=stdout)
(src_train_sents, tgt_train_sents,
    src_dev_sents, tgt_dev_sents,
    src_test_sents, tgt_test_sents) = get_data(data_path, args.language)


# Build a vocabulary of source and target language.
vocab_file = "vocab_%s_en.json" % args.language
vocab = Vocab.build(src_train_sents, tgt_train_sents, args.vocab_size, args.freq_cutoff)
vocab.save(vocab_file)


# Build a model object.
model = NMT(word_embed_size=args.word_embed_size, char_embed_size=args.char_embed_size,
            hidden_size=args.hidden_size, vocab=vocab, dropout_rate=args.dropout_rate,
            kernel_size=args.kernel_size, padding=args.padding)


# Train the model.
train_data = list(zip(src_train_sents, tgt_train_sents))
dev_data = list(zip(src_dev_sents, tgt_dev_sents))
dataset = {"train_data" : train_data, "dev_data" : dev_data}
model_save_path = args.destination + "bin/model_%s_en.bin" % args.language
tic = time.time()
train(model, dataset, learning_rate=args.learning_rate, lr_decay=args.lr_decay,
        clip_grad=args.clip_grad, batch_size=args.batch_size, max_epochs=args.max_epochs,
        max_num_trial=args.max_num_trial, patience_limit=args.patience_limit,
        model_save_path=model_save_path, stdout=stdout)
toc = time.time()
print(f"Training took {((toc - tic) / 60):.3f} minutes", file=stdout)


# Compute and print BLEU score.
print("Computing corpuse level BLEU score..", file=stdout)
test_data = [src_test_sents, tgt_test_sents]
tic = time.time()
bleu_score = compute_corpus_level_bleu_score(model=model, data=test_data)
toc = time.time()
print(f"Corpus BLEU: {bleu_score*100:.3f}. Computed in {toc-tic:.3f} seconds.", file=stdout)


# Close the logging file.
stdout.close()

#