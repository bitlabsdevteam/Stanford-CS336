"""Generate beginner-friendly Colab notebooks for 5D GPU parallelism concepts.

The notebooks are designed to run in a single Python runtime, including CPU-only
Colab sessions. Each notebook focuses on one parallelism dimension and uses
small, readable PyTorch examples that simulate the distributed idea.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "5D_Parallelism"


def to_source(text: str) -> list[str]:
    """Convert a block of text into the list-of-lines notebook format."""
    return dedent(text).strip("\n").splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    """Build a markdown notebook cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source(text),
    }


def code_cell(text: str) -> dict:
    """Build a code notebook cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(text),
    }


COMMON_BOOTSTRAP = """
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("torch") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(7)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
"""


NOTEBOOKS: list[dict] = [
    {
        "filename": "1_DataParallelism.ipynb",
        "title": "1. Data Parallelism",
        "cells": [
            markdown_cell(
                """
                # 1. Data Parallelism

                This notebook introduces the simplest training parallelism first.

                **Core idea**
                - every GPU keeps the same model
                - the batch is split into smaller pieces
                - each GPU computes gradients on its own data shard
                - gradients are averaged before the optimizer step

                In a real multi-GPU run, each model copy lives on a different GPU.
                In this notebook, we simulate the same math on one runtime so the
                idea is easy to see in Colab.
                """
            ),
            code_cell(COMMON_BOOTSTRAP),
            markdown_cell(
                """
                ## Why start here?

                Data parallelism is usually the easiest parallelism to understand
                because the model itself does not change. Only the **batch**
                changes.

                We will compare:
                1. one normal full-batch training step
                2. one virtual data-parallel training step with two replicas

                If both are implemented correctly, the parameter updates should be
                almost identical.
                """
            ),
            code_cell(
                """
                from copy import deepcopy


                class TinyClassifier(nn.Module):
                    \"\"\"Small network used to demonstrate gradient averaging.\"\"\"

                    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, num_classes),
                        )

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return self.net(x)


                batch_size = 8
                input_dim = 4
                hidden_dim = 10
                num_classes = 3
                learning_rate = 0.1

                x = torch.randn(batch_size, input_dim)
                y = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0], dtype=torch.long)

                base_model = TinyClassifier(input_dim, hidden_dim, num_classes)
                base_state = deepcopy(base_model.state_dict())
                criterion = nn.CrossEntropyLoss(reduction="sum")

                print("Input batch shape:", tuple(x.shape))
                print("Labels shape:", tuple(y.shape))
                """
            ),
            code_cell(
                """
                def run_full_batch_step(initial_state: dict[str, torch.Tensor]) -> tuple[float, dict[str, torch.Tensor]]:
                    model = TinyClassifier(input_dim, hidden_dim, num_classes)
                    model.load_state_dict(initial_state)
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

                    optimizer.zero_grad()
                    logits = model(x)
                    loss = criterion(logits, y) / batch_size
                    loss.backward()
                    optimizer.step()

                    return loss.item(), deepcopy(model.state_dict())


                def run_virtual_data_parallel_step(
                    initial_state: dict[str, torch.Tensor],
                    num_replicas: int,
                ) -> tuple[float, dict[str, torch.Tensor]]:
                    reference_model = TinyClassifier(input_dim, hidden_dim, num_classes)
                    reference_model.load_state_dict(initial_state)
                    optimizer = torch.optim.SGD(reference_model.parameters(), lr=learning_rate)

                    for parameter in reference_model.parameters():
                        parameter.grad = torch.zeros_like(parameter)

                    total_loss = 0.0
                    x_shards = x.chunk(num_replicas)
                    y_shards = y.chunk(num_replicas)

                    for replica_id, (x_shard, y_shard) in enumerate(zip(x_shards, y_shards)):
                        replica = TinyClassifier(input_dim, hidden_dim, num_classes)
                        replica.load_state_dict(initial_state)

                        shard_logits = replica(x_shard)
                        shard_loss_sum = criterion(shard_logits, y_shard)
                        shard_loss_sum.backward()
                        total_loss += shard_loss_sum.item()

                        print(
                            f"Replica {replica_id}: shard shape={tuple(x_shard.shape)}, "
                            f"examples={len(x_shard)}"
                        )

                        for reference_parameter, replica_parameter in zip(
                            reference_model.parameters(),
                            replica.parameters(),
                        ):
                            reference_parameter.grad += replica_parameter.grad

                    for parameter in reference_model.parameters():
                        parameter.grad /= batch_size

                    optimizer.step()
                    mean_loss = total_loss / batch_size
                    return mean_loss, deepcopy(reference_model.state_dict())
                """
            ),
            code_cell(
                """
                full_loss, full_state = run_full_batch_step(base_state)
                dp_loss, dp_state = run_virtual_data_parallel_step(base_state, num_replicas=2)


                def max_state_difference(
                    left: dict[str, torch.Tensor],
                    right: dict[str, torch.Tensor],
                ) -> float:
                    differences = [
                        (left[name] - right[name]).abs().max().item()
                        for name in left
                    ]
                    return max(differences)


                print()
                print(f"Full-batch loss:         {full_loss:.6f}")
                print(f"Virtual data-parallel:   {dp_loss:.6f}")
                print(f"Max parameter difference: {max_state_difference(full_state, dp_state):.10f}")

                assert torch.isclose(torch.tensor(full_loss), torch.tensor(dp_loss), atol=1e-7)
                assert max_state_difference(full_state, dp_state) < 1e-7

                print("The data-parallel update matches the full-batch update.")
                """
            ),
            markdown_cell(
                """
                ## Takeaways

                - Data parallelism keeps the model architecture unchanged.
                - The main communication step is **gradient averaging**.
                - This is often the first parallelism to try when one GPU is too slow
                  but the model still fits in memory.
                """
            ),
        ],
    },
    {
        "filename": "2_PipelineParallelism.ipynb",
        "title": "2. Pipeline Parallelism",
        "cells": [
            markdown_cell(
                """
                # 2. Pipeline Parallelism

                Pipeline parallelism splits the model into **stages**.

                **Core idea**
                - GPU 0 runs the early layers
                - GPU 1 runs the later layers
                - the batch is broken into micro-batches
                - each micro-batch flows through the stages like an assembly line

                This helps when a single GPU cannot hold the whole model.
                """
            ),
            code_cell(COMMON_BOOTSTRAP),
            markdown_cell(
                """
                ## Beginner mental model

                Think of a two-stage factory:
                - Stage 1 prepares features
                - Stage 2 turns those features into predictions

                We will:
                1. split a tiny model into two stages
                2. run the batch in micro-batches
                3. verify that the pipelined result matches the full model
                """
            ),
            code_cell(
                """
                from copy import deepcopy


                class Stage1(nn.Module):
                    \"\"\"First half of the model.\"\"\"

                    def __init__(self, input_dim: int, hidden_dim: int) -> None:
                        super().__init__()
                        self.linear = nn.Linear(input_dim, hidden_dim)
                        self.activation = nn.ReLU()

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return self.activation(self.linear(x))


                class Stage2(nn.Module):
                    \"\"\"Second half of the model.\"\"\"

                    def __init__(self, hidden_dim: int, output_dim: int) -> None:
                        super().__init__()
                        self.linear = nn.Linear(hidden_dim, output_dim)

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return self.linear(x)


                class TinyPipelineModel(nn.Module):
                    \"\"\"Convenient full model for comparison.\"\"\"

                    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
                        super().__init__()
                        self.stage1 = Stage1(input_dim, hidden_dim)
                        self.stage2 = Stage2(hidden_dim, output_dim)

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return self.stage2(self.stage1(x))


                input_dim = 6
                hidden_dim = 8
                output_dim = 2
                batch_size = 8
                micro_batch_size = 2
                learning_rate = 0.05

                x = torch.randn(batch_size, input_dim)
                y = torch.randn(batch_size, output_dim)
                criterion = nn.MSELoss(reduction="sum")

                base_model = TinyPipelineModel(input_dim, hidden_dim, output_dim)
                base_state = deepcopy(base_model.state_dict())

                print("Batch shape:", tuple(x.shape))
                print("Micro-batches:", len(x) // micro_batch_size)
                """
            ),
            code_cell(
                """
                full_model = TinyPipelineModel(input_dim, hidden_dim, output_dim)
                full_model.load_state_dict(base_state)
                full_output = full_model(x)

                stage1 = deepcopy(full_model.stage1)
                stage2 = deepcopy(full_model.stage2)

                pipeline_outputs = []
                for micro_batch_id, x_micro in enumerate(x.split(micro_batch_size)):
                    hidden = stage1(x_micro)
                    output = stage2(hidden)
                    pipeline_outputs.append(output)
                    print(
                        f"Micro-batch {micro_batch_id}: "
                        f"{tuple(x_micro.shape)} -> {tuple(hidden.shape)} -> {tuple(output.shape)}"
                    )

                pipeline_output = torch.cat(pipeline_outputs, dim=0)

                print()
                print("Max forward difference:", (full_output - pipeline_output).abs().max().item())
                assert torch.allclose(full_output, pipeline_output, atol=1e-7)
                print("Forward pass matches the unsplit model.")
                """
            ),
            code_cell(
                """
                def train_full_model(initial_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                    model = TinyPipelineModel(input_dim, hidden_dim, output_dim)
                    model.load_state_dict(initial_state)
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

                    optimizer.zero_grad()
                    loss = criterion(model(x), y) / batch_size
                    loss.backward()
                    optimizer.step()
                    return deepcopy(model.state_dict())


                def train_pipeline_model(initial_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                    model = TinyPipelineModel(input_dim, hidden_dim, output_dim)
                    model.load_state_dict(initial_state)
                    stage1 = deepcopy(model.stage1)
                    stage2 = deepcopy(model.stage2)
                    optimizer = torch.optim.SGD(
                        list(stage1.parameters()) + list(stage2.parameters()),
                        lr=learning_rate,
                    )

                    optimizer.zero_grad()

                    for x_micro, y_micro in zip(x.split(micro_batch_size), y.split(micro_batch_size)):
                        prediction = stage2(stage1(x_micro))
                        micro_loss = criterion(prediction, y_micro)
                        micro_loss.backward()

                    for parameter in list(stage1.parameters()) + list(stage2.parameters()):
                        parameter.grad /= batch_size

                    optimizer.step()

                    merged = TinyPipelineModel(input_dim, hidden_dim, output_dim)
                    merged.stage1.load_state_dict(stage1.state_dict())
                    merged.stage2.load_state_dict(stage2.state_dict())
                    return deepcopy(merged.state_dict())


                full_state = train_full_model(base_state)
                pipeline_state = train_pipeline_model(base_state)

                max_difference = max(
                    (full_state[name] - pipeline_state[name]).abs().max().item()
                    for name in full_state
                )

                print("Max parameter difference after one step:", max_difference)
                assert max_difference < 1e-7
                print("Pipeline training matches the unsplit training step.")
                """
            ),
            markdown_cell(
                """
                ## Takeaways

                - Pipeline parallelism splits the model by **layers or blocks**.
                - Micro-batches keep later stages busy instead of waiting for the
                  whole batch.
                - The main tradeoff is extra scheduling complexity and possible
                  idle time, sometimes called a **pipeline bubble**.
                """
            ),
        ],
    },
    {
        "filename": "3_TensorParallelism.ipynb",
        "title": "3. Tensor Parallelism",
        "cells": [
            markdown_cell(
                """
                # 3. Tensor Parallelism

                Tensor parallelism splits a **single large layer** across multiple GPUs.

                **Core idea**
                - each GPU stores only part of one weight matrix
                - every GPU computes part of the same layer
                - partial results are combined into the final answer

                This is more advanced than data or pipeline parallelism because a
                single layer is now distributed.
                """
            ),
            code_cell(COMMON_BOOTSTRAP),
            markdown_cell(
                """
                ## Two common ways to split a linear layer

                1. **Column parallelism**
                   Each GPU produces a slice of the output features.
                2. **Row parallelism**
                   Each GPU handles a slice of the input features, and the partial
                   outputs are added together.

                We will show both with one small `nn.Linear` layer.
                """
            ),
            code_cell(
                """
                in_features = 6
                out_features = 8
                batch_size = 3
                num_shards = 2

                linear = nn.Linear(in_features, out_features)
                x = torch.randn(batch_size, in_features)
                full_output = linear(x)

                print("Input shape:", tuple(x.shape))
                print("Weight shape:", tuple(linear.weight.shape))
                print("Full output shape:", tuple(full_output.shape))
                """
            ),
            code_cell(
                """
                # Column parallelism: split output features across shards.
                column_weight_shards = linear.weight.chunk(num_shards, dim=0)
                column_bias_shards = linear.bias.chunk(num_shards, dim=0)

                column_outputs = []
                for shard_id, (weight_shard, bias_shard) in enumerate(
                    zip(column_weight_shards, column_bias_shards)
                ):
                    local_output = F.linear(x, weight_shard, bias_shard)
                    column_outputs.append(local_output)
                    print(
                        f"Column shard {shard_id}: "
                        f"weight={tuple(weight_shard.shape)}, output={tuple(local_output.shape)}"
                    )

                column_parallel_output = torch.cat(column_outputs, dim=-1)

                print()
                print(
                    "Max difference after column parallelism:",
                    (full_output - column_parallel_output).abs().max().item(),
                )
                assert torch.allclose(full_output, column_parallel_output, atol=1e-7)
                """
            ),
            code_cell(
                """
                # Row parallelism: split input features across shards.
                x_shards = x.chunk(num_shards, dim=-1)
                row_weight_shards = linear.weight.chunk(num_shards, dim=1)

                partial_outputs = []
                for shard_id, (x_shard, weight_shard) in enumerate(zip(x_shards, row_weight_shards)):
                    local_partial = F.linear(x_shard, weight_shard, bias=None)
                    partial_outputs.append(local_partial)
                    print(
                        f"Row shard {shard_id}: "
                        f"input={tuple(x_shard.shape)}, weight={tuple(weight_shard.shape)}"
                    )

                row_parallel_output = sum(partial_outputs) + linear.bias

                print()
                print(
                    "Max difference after row parallelism:",
                    (full_output - row_parallel_output).abs().max().item(),
                )
                assert torch.allclose(full_output, row_parallel_output, atol=1e-7)
                print("Both tensor-parallel splits reproduce the full layer output.")
                """
            ),
            markdown_cell(
                """
                ## Takeaways

                - Tensor parallelism is useful when a single layer is too large for
                  one GPU.
                - Column parallelism combines outputs with **concatenation**.
                - Row parallelism combines outputs with **summation**.
                - This usually needs more communication than pure data parallelism.
                """
            ),
        ],
    },
    {
        "filename": "4_ContextParallelism.ipynb",
        "title": "4. Context Parallelism",
        "cells": [
            markdown_cell(
                """
                # 4. Context Parallelism

                Context parallelism splits a long sequence across GPUs along the
                **token dimension**.

                **Core idea**
                - GPU 0 owns one slice of the sequence
                - GPU 1 owns another slice
                - each GPU produces outputs for its own tokens
                - attention still needs information from the full context

                This notebook shows the basic idea with a tiny single-head attention
                block.
                """
            ),
            code_cell(COMMON_BOOTSTRAP),
            markdown_cell(
                """
                ## Why this is more advanced

                With context parallelism, the sequence is split, but each token may
                still need to look at other tokens.

                That means:
                - some operations are local and easy
                - attention usually needs communication so every shard can use the
                  full set of keys and values

                We simulate that communication by gathering keys and values from all
                shards before each local attention computation.
                """
            ),
            code_cell(
                """
                import math


                class TinyAttention(nn.Module):
                    \"\"\"Single-head attention kept intentionally small and readable.\"\"\"

                    def __init__(self, hidden_size: int) -> None:
                        super().__init__()
                        self.hidden_size = hidden_size
                        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

                    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

                    def attend(
                        self,
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                    ) -> torch.Tensor:
                        scores = q @ k.transpose(-2, -1) / math.sqrt(self.hidden_size)
                        weights = scores.softmax(dim=-1)
                        return weights @ v

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        q, k, v = self.project(x)
                        context = self.attend(q, k, v)
                        return self.out_proj(context)


                hidden_size = 4
                sequence_length = 6
                num_shards = 2

                attention = TinyAttention(hidden_size)
                x = torch.randn(1, sequence_length, hidden_size)

                full_output = attention(x)

                print("Input shape:", tuple(x.shape))
                print("Full attention output shape:", tuple(full_output.shape))
                """
            ),
            code_cell(
                """
                x_shards = x.chunk(num_shards, dim=1)
                local_qkv = [attention.project(shard) for shard in x_shards]

                all_k = torch.cat([k for _, k, _ in local_qkv], dim=1)
                all_v = torch.cat([v for _, _, v in local_qkv], dim=1)

                local_outputs = []
                for shard_id, (q_local, _, _) in enumerate(local_qkv):
                    local_context = attention.attend(q_local, all_k, all_v)
                    local_output = attention.out_proj(local_context)
                    local_outputs.append(local_output)
                    print(
                        f"Shard {shard_id}: "
                        f"local queries={tuple(q_local.shape)}, "
                        f"global keys={tuple(all_k.shape)}"
                    )

                context_parallel_output = torch.cat(local_outputs, dim=1)

                print()
                print(
                    "Max difference after context parallelism:",
                    (full_output - context_parallel_output).abs().max().item(),
                )
                assert torch.allclose(full_output, context_parallel_output, atol=1e-7)
                print("Context-parallel attention matches the full attention output.")
                """
            ),
            markdown_cell(
                """
                ## Takeaways

                - Context parallelism helps when the **sequence length** is the main
                  memory problem.
                - Each shard owns only part of the tokens.
                - Attention often requires communication because local queries may
                  need global keys and values.
                - This idea becomes very useful for long-context language models.
                """
            ),
        ],
    },
    {
        "filename": "5_ExpertParallelism.ipynb",
        "title": "5. Expert Parallelism",
        "cells": [
            markdown_cell(
                """
                # 5. Expert Parallelism

                Expert parallelism is commonly used in **Mixture-of-Experts (MoE)**
                models.

                **Core idea**
                - the model has several experts
                - a router decides which expert should process each token
                - different experts can live on different GPUs
                - only a subset of experts runs for each token

                This is the most advanced notebook in the set because it adds
                **conditional computation** on top of parallelism.
                """
            ),
            code_cell(COMMON_BOOTSTRAP),
            markdown_cell(
                """
                ## Beginner mental model

                Imagine a team with specialists:
                - one expert is good at one type of token
                - another expert is good at another type
                - the router sends each token to one specialist

                The big win is that the whole model can have many experts, while a
                single token only uses a small part of them.
                """
            ),
            code_cell(
                """
                class Expert(nn.Module):
                    \"\"\"A tiny expert network.\"\"\"

                    def __init__(self, hidden_size: int) -> None:
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size),
                        )

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return self.net(x)


                class Top1MoE(nn.Module):
                    \"\"\"Routes each token to exactly one expert.\"\"\"

                    def __init__(self, hidden_size: int, num_experts: int) -> None:
                        super().__init__()
                        self.router = nn.Linear(hidden_size, num_experts, bias=False)
                        self.experts = nn.ModuleList([Expert(hidden_size) for _ in range(num_experts)])

                    def forward(
                        self,
                        tokens: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
                        router_scores = self.router(tokens)
                        chosen_experts = router_scores.argmax(dim=-1)
                        outputs = torch.zeros_like(tokens)
                        load_per_expert: dict[int, int] = {}

                        for expert_id, expert in enumerate(self.experts):
                            mask = chosen_experts == expert_id
                            load_per_expert[expert_id] = int(mask.sum().item())
                            if mask.any():
                                outputs[mask] = expert(tokens[mask])

                        return outputs, chosen_experts, load_per_expert


                num_tokens = 8
                hidden_size = 4
                num_experts = 3

                tokens = torch.randn(num_tokens, hidden_size)
                moe = Top1MoE(hidden_size, num_experts)

                expert_outputs, expert_ids, load_per_expert = moe(tokens)
                final_output = tokens + 0.3 * expert_outputs

                print("Token shape:", tuple(tokens.shape))
                print("Chosen expert per token:", expert_ids.tolist())
                print("Load per expert:", load_per_expert)
                print("Final output shape:", tuple(final_output.shape))
                """
            ),
            code_cell(
                """
                # Show exactly which tokens were sent to each expert.
                for expert_id in range(num_experts):
                    token_positions = (expert_ids == expert_id).nonzero(as_tuple=False).squeeze(-1).tolist()
                    print(f"Expert {expert_id} receives token positions: {token_positions}")

                assert expert_outputs.shape == tokens.shape
                assert final_output.shape == tokens.shape
                assert sum(load_per_expert.values()) == num_tokens

                print()
                print("Every token was routed to one expert, and the output shape is correct.")
                """
            ),
            markdown_cell(
                """
                ## Takeaways

                - Expert parallelism distributes **experts**, not just tensors or
                  batches.
                - The router is responsible for sending tokens to experts.
                - MoE models can increase model capacity without forcing every token
                  to use the full network.
                - Real systems also worry about load balancing so one expert does
                  not receive almost everything.
                """
            ),
        ],
    },
]


def notebook_metadata(title: str) -> dict:
    """Return lightweight Colab-friendly notebook metadata."""
    return {
        "colab": {
            "name": title,
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }


def write_notebook(filename: str, title: str, cells: list[dict]) -> None:
    """Write one notebook to disk."""
    notebook = {
        "cells": cells,
        "metadata": notebook_metadata(title),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path = OUTPUT_DIR / filename
    output_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def write_readme() -> None:
    """Write a short index file that explains the notebook order."""
    readme_text = dedent(
        """
        # 5D Parallelism Notebook Set

        The notebooks are ordered from easier to more advanced:

        1. `1_DataParallelism.ipynb`
        2. `2_PipelineParallelism.ipynb`
        3. `3_TensorParallelism.ipynb`
        4. `4_ContextParallelism.ipynb`
        5. `5_ExpertParallelism.ipynb`

        Each notebook is self-contained, uses Python with PyTorch, and keeps the
        examples small enough for a beginner to follow in Colab.
        """
    ).strip() + "\n"
    (OUTPUT_DIR / "README.md").write_text(readme_text, encoding="utf-8")


def main() -> None:
    """Generate all notebooks and the index file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for notebook in NOTEBOOKS:
        write_notebook(
            filename=notebook["filename"],
            title=notebook["title"],
            cells=notebook["cells"],
        )
    write_readme()
    print(f"Wrote {len(NOTEBOOKS)} notebooks to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
