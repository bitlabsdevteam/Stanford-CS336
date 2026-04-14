# Stanford CS336 Assignment Workspace

This repository contains the notebook and module work for a Stanford CS336-style language modeling assignment. The GPT implementation lives in [`GPT/`](/Users/davidbong/Documents/LLM_Assignments/GPT), the training entrypoint lives in [`scripts/train_transformer_lm.py`](/Users/davidbong/Documents/LLM_Assignments/scripts/train_transformer_lm.py), and the current automated checks live in [`tests/`](/Users/davidbong/Documents/LLM_Assignments/tests).

## Setup

Create the notebook-friendly virtual environment:

```bash
bash scripts/setup_notebook_env.sh
source .venv/bin/activate
```

If you already manage your own environment, make sure it includes at least `numpy`, `torch`, and `pytest`.

## Run Tests

Run the full test suite:

```bash
pytest
```

Run a single test file:

```bash
pytest tests/test_transformer_lm.py
```

Run one specific test:

```bash
pytest tests/test_transformer_lm.py -k tied_head
```

Check Python syntax for the training code:

```bash
python -m py_compile scripts/train_transformer_lm.py GPT/*.py
```

The current test suite covers the core GPT building blocks, including embeddings, RMSNorm, attention, multi-head attention, Transformer blocks, decoding, optimization, and the full Transformer LM.

## Benchmark The GPT Model

Use [`scripts/benchmark_transformer.py`](/Users/davidbong/Documents/LLM_Assignments/scripts/benchmark_transformer.py) to time forward-only, forward-plus-backward, or full training-step passes with random weights and random token batches:

```bash
python scripts/benchmark_transformer.py \
  --model-size small \
  --context-length 128 \
  --batch-size 4 \
  --warmup-steps 5 \
  --measure-steps 10 \
  --mode forward-backward \
  --device cuda \
  --dtype float32
```

Useful options:

- `--mode forward` benchmarks only the forward pass.
- `--mode forward-backward` benchmarks a training-style forward pass, cross-entropy loss, and backward pass.
- `--mode train-step` benchmarks forward pass, loss, backward pass, and the assignment `AdamW` optimizer step.
- `--mixed-precision-dtype bf16` runs the forward and loss under autocast while keeping model parameters in `--dtype`.
- `--model-size` accepts `small`, `medium`, `large`, `xl`, and `2.7b`.
- `--d-model`, `--d-ff`, `--num-layers`, and `--num-heads` can override a preset for quick smoke tests.
- `--json` prints machine-readable output for sweep scripts or notebook tables.

For Nsight Systems profiling on a CUDA machine, the benchmark emits NVTX ranges for warmup, measured iterations, forward, backward, optimizer step, and the major self-attention sub-operations. Example:

```bash
uv run nsys profile -o nsys_small_ctx128 \
  python scripts/benchmark_transformer.py \
  --model-size small \
  --context-length 128 \
  --batch-size 4 \
  --warmup-steps 5 \
  --measure-steps 1 \
  --mode train-step \
  --device cuda \
  --dtype float32
```

Example BF16 mixed-precision benchmark:

```bash
python scripts/benchmark_transformer.py \
  --model-size small \
  --context-length 128 \
  --batch-size 4 \
  --warmup-steps 5 \
  --measure-steps 10 \
  --mode forward-backward \
  --device cuda \
  --dtype float32 \
  --mixed-precision-dtype bf16
```

Example CUDA memory snapshot for the PyTorch memory visualizer:

```bash
python scripts/benchmark_transformer.py \
  --model-size 2.7b \
  --context-length 128 \
  --batch-size 1 \
  --warmup-steps 3 \
  --measure-steps 1 \
  --mode train-step \
  --device cuda \
  --dtype float32 \
  --memory-profile-path memory_snapshot.pickle
```

When running on CUDA, the benchmark also reports `peak_memory_allocated_mib` and `peak_memory_reserved_mib`. If `--memory-profile-path` is set, it records allocation history after warmup, dumps a pickle file at the end of the measured region, and that file can be loaded into the PyTorch memory visualizer at `pytorch.org/memory_viz`.

To compare eager vs. `torch.compile` on the same Transformer configuration, pass both execution variants:

```bash
python scripts/benchmark_transformer.py \
  --model-size small \
  --context-length 128 \
  --batch-size 4 \
  --warmup-steps 5 \
  --measure-steps 10 \
  --mode train-step \
  --device cuda \
  --execution-variants eager,compiled
```

## Benchmark Attention

Use [`scripts/benchmark_attention.py`](/Users/davidbong/Documents/LLM_Assignments/scripts/benchmark_attention.py) to sweep the standalone scaled dot-product attention kernel over the assignment grid of head dimensions and sequence lengths:

```bash
python scripts/benchmark_attention.py \
  --device cuda \
  --dtype float32 \
  --warmup-steps 10 \
  --iterations 100 \
  --variants eager,compiled
```

The default sweep matches the assignment: batch size `8`, head dimensions `16,32,64,128`, and sequence lengths `256,1024,4096,8192,16384`. The markdown table reports forward and backward timings, the CUDA memory allocated immediately before backward, the quadratic `(B, T, T)` tensor footprint, and any out-of-memory stage.

## Benchmark Distributed All-Reduce

Use [`scripts/benchmark_distributed_all_reduce.py`](/Users/davidbong/Documents/LLM_Assignments/scripts/benchmark_distributed_all_reduce.py) to benchmark single-node multi-process `all_reduce` over the assignment communication grid:

```bash
python scripts/benchmark_distributed_all_reduce.py \
  --backend-device-pairs gloo+cpu,nccl+cuda \
  --data-sizes-mb 1,10,100,1024 \
  --process-counts 2,4,6 \
  --warmup-iterations 5 \
  --measure-iterations 20
```

The script prints a markdown table with per-configuration latency summaries and a short commentary block. On CPU-only machines, the `nccl+cuda` rows are reported as skipped; on multi-GPU machines, ensure at least as many visible GPUs as the requested process count.

## Train The GPT Model

Use the training CLI in [`scripts/train_transformer_lm.py`](/Users/davidbong/Documents/LLM_Assignments/scripts/train_transformer_lm.py):

```bash
python scripts/train_transformer_lm.py \
  --train-data path/to/train_tokens.npy \
  --val-data path/to/val_tokens.npy \
  --vocab-size 10000 \
  --context-length 128 \
  --num-layers 4 \
  --d-model 256 \
  --num-heads 4 \
  --d-ff 1024 \
  --batch-size 16 \
  --total-iters 1000 \
  --learning-rate 3e-4 \
  --checkpoint checkpoints/gpt.pt
```

### Training data format

- `--train-data` is required.
- `--val-data` is optional but recommended.
- Each data file must be a one-dimensional token array.
- Supported formats are:
  - `.npy` arrays loaded with memory mapping
  - raw contiguous token buffers opened as memmaps
- The default raw token dtype is `uint16`. Override it with `--train-data-dtype` or `--val-data-dtype` if your token files use a different integer type.

### Useful options

- `--device auto` chooses `cuda`, then `mps`, then `cpu`.
- `--dtype` accepts values such as `float32`, `float16`, and `bfloat16`.
- `--eval-interval` controls how often train and validation loss are reported.
- `--checkpoint-interval` controls how often the checkpoint file is updated.
- `--resume path/to/checkpoint.pt` resumes training from a saved checkpoint.
- `--log-interval` controls how often a training log line is printed.

Example resume command:

```bash
python scripts/train_transformer_lm.py \
  --train-data path/to/train_tokens.npy \
  --val-data path/to/val_tokens.npy \
  --vocab-size 10000 \
  --checkpoint checkpoints/gpt.pt \
  --resume checkpoints/gpt.pt
```

### Notes

- `--vocab-size` must match the tokenizer that produced your token IDs.
- `context_length` must be smaller than the number of available tokens in each dataset split.
- Checkpoints store the model state, optimizer state, and iteration number.
