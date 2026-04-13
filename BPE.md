# BPE Tokenizer Testing Guide

## Overview
The production BPE implementation now lives in the `bpe_tokenizer/` package. Use it for training, encoding, decoding, benchmarking, and dataset serialization. The main CLI entrypoint is `python -m bpe_tokenizer.cli`.

## 1. Basic Sanity Check
Run a syntax check first:

```bash
python -m py_compile bpe_tokenizer/*.py
```

This confirms the package imports cleanly before longer runs.

## 2. Train a Tokenizer
Train TinyStories with `<|endoftext|>` as a hard boundary:

```bash
python -m bpe_tokenizer.cli train \
  --input /path/to/TinyStories.txt \
  --vocab-size 10000 \
  --special-token '<|endoftext|>' \
  --num-processes 8 \
  --vocab-out tinystories_vocab.json \
  --merges-out tinystories_merges.txt
```

The command prints runtime, RSS memory, longest token, and a compact profile summary.

## 3. Benchmark Compression and Throughput
Sample documents and measure bytes per token and bytes per second:

```bash
python -m bpe_tokenizer.cli benchmark \
  --vocab tinystories_vocab.json \
  --merges tinystories_merges.txt \
  --special-token '<|endoftext|>' \
  --input /path/to/TinyStories.txt \
  --document-delimiter '<|endoftext|>' \
  --num-documents 10 \
  --repeats 3 \
  --dataset-bytes 825000000000
```

Use the same command on OpenWebText with its own vocab and merges.

## 4. Test Streaming Encoding
Encode a large corpus to `uint16` token IDs:

```bash
python -m bpe_tokenizer.cli encode \
  --vocab tinystories_vocab.json \
  --merges tinystories_merges.txt \
  --special-token '<|endoftext|>' \
  --input /path/to/TinyStories.txt \
  --output tinystories_tokens.npy
```

The output `.npy` file should load as `uint16`, which is correct for vocabularies below 65,536 IDs.

## 5. Assignment Tests
If you have the CS336 tests locally, run:

```bash
pytest tests/test_train_bpe.py
pytest tests/test_tokenizer.py
```

You will still need to point the assignment adapters at `bpe_tokenizer.trainer.train_bpe` and `bpe_tokenizer.tokenizer.Tokenizer`.
