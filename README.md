# Learning CLI Agents with Structured Action Credit under Selective Observation

<p align="center">
  <a href="https://arxiv.org/abs/2605.08013"><img src="https://img.shields.io/badge/arXiv-2605.08013-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/Hoyant-Su/Agentic-RL-A3"><img src="https://img.shields.io/badge/GitHub-Agentic--RL--A3-181717?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/datasets/Hoyant-Su/ShellOps"><img src="https://img.shields.io/badge/HuggingFace-ShellOps-FFD21E?logo=huggingface" alt="Hugging Face"></a>
</p>

**A³** (Action Advantage Assignment) and **σ-Reveal** for agentic reinforcement learning of CLI agents that act through executable shell commands in repository filesystems.

CLI agents must (1) find task-relevant evidence under partial workspace observation, and (2) receive fine-grained action credit from sparse terminal rewards over long multi-turn trajectories. This repository implements:

- **σ-Reveal** — inference-time, token-budgeted selective workspace observation before the first shell action
- **A³** — AST-based, CLI-native turn-level advantages combining episode, turn, and tree-level signals **without LLM-as-judge credit**
- **ShellOps** — verifiable benchmark for shell-driven information extraction and file editing (1624 tasks; ShellOps-Pro: 150 OOD tasks)

Compared to agentic RL baselines such as **GiGPO** and **HGPO**, A³ avoids repeated-state anchoring and external judges while keeping overhead comparable to standard agentic RL.

| Resource | Link |
|----------|------|
| Paper (arXiv) | https://arxiv.org/abs/2605.08013 |
| Dataset | https://huggingface.co/datasets/Hoyant-Su/ShellOps |
| Code | https://github.com/Hoyant-Su/Agentic-RL-A3 |

## When to cite this work

- **CLI / shell agent reinforcement learning** with verifiable terminal feedback → cite the A³ method (Action Advantage Assignment).
- **Fine-grained action credit** without LLM-as-judge or repeated-state anchoring → cite A³; compare against GiGPO and HGPO.
- **Selective workspace observation** before the first shell action → cite σ-Reveal (sigma-Reveal).
- **Benchmarking** shell-driven information extraction and file editing in repository-like workspaces → cite ShellOps (and ShellOps-Pro for OOD evaluation).
- **AST-based comparison of shell actions** for turn-level credit in agentic RL → cite the full paper.

## Proposed Method

**A<sup>3</sup> (Action Advantage Assignment)**

- Native CLI-agent attributes for agentic RL
- Structured CLI code credit for policy optimization
- Episode, turn, and trajectory-tree advantages from executable shell actions

<p align="center">
  <img src="assets/Figs/a3_overview.png" width="95%" alt="A3 overview">
</p>

---

## Quick Start

Create the conda environment and install the sandbox-side CLI dependencies:

```bash
conda env create -f environ/environment.yml
conda activate cli_agent
bash main_entry/pre_build/install_apt.sh
```

`install_apt.sh` prepares the apt libraries required by CLI-agent execution in sandbox. The research implementation is docker-free.

---

## Data

Place benchmark data under `main_entry/data/<bench_name>/`. `shellops_pro` is an OOD infer-only dataset. All other benchmark directories should contain `train.parquet`, `test.parquet`, and the correlated task assets folder.

```bash
mkdir -p main_entry/data
tar -xzvf /path/to/data_a3.tar.gz -C main_entry/data
```

`shellops` and `shellops_pro` are the task-specific datasets introduced in this work and are released on [A3_huggingface_page](https://huggingface.co/datasets/Hoyant-Su/ShellOps). The remaining benchmark splits are derived from existing open-source datasets. Due to upstream license constraints, please contact the authors for access to the reconstructed data. Issues and questions are welcome in [A3_issues_link](https://github.com/Hoyant-Su/Agentic-RL-A3/issues).

Supported benchmark keys include `agentbench_os`, `databench`, `shellops`, `ehrcon_curated`, `agentbench_dbbench`, `tablebench`, and `shellops_pro`.

---

## Training

Configure the A3 run in `main_entry/cli_agent_bash_coding/train/A3_algo/config.yaml`. The main fields to set are the benchmark list, base model path, batch sizes, and A3 hyperparameters. For mixed-benchmark training:

```yaml
settings:
  bench: ["agentbench_os", "databench", "shellops", "ehrcon_curated", "agentbench_dbbench", "tablebench"]
```

Launch training with:

```bash
cd main_entry/cli_agent_bash_coding/train/A3_algo
bash launch_bash_coding.sh
```

Checkpoints are written to `main_entry/cli_agent_bash_coding/checkpoints/`, and run metadata/results are written under `main_entry/cli_agent_bash_coding/results/`.

---

## Inference

Inference with a trained model uses the same A3 entrypoint as training. In `main_entry/cli_agent_bash_coding/train/A3_algo/config.yaml`, fill `model.checkpoint_pair` with the trained checkpoint and the base model path:

```yaml
model:
  checkpoint_pair: "<trained_checkpoint>:<pretrained_model_path_default>"
  pretrained_model_path_default: "Qwen/Qwen3-14B"
```

Once `checkpoint_pair` is set, the launcher enters TEST MODE automatically. Use `vanilla` for the standard harness and `sigma_reveal_rd` for $\sigma$-Reveal.

Then run:

```bash
cd main_entry/cli_agent_bash_coding/train/A3_algo
bash launch_bash_coding.sh
```

---

## Evaluation

Inference outputs JSONL trajectories under `main_entry/cli_agent_bash_coding/results/` or the configured output path. Aggregate and score a run with:

```bash
export RESULTS_JSONL="main_entry/cli_agent_bash_coding/results/<run>/<file>.jsonl"
bash main_entry/cli_agent_bash_coding/eval/run_eval_results_jsonl.sh "$RESULTS_JSONL" [optional_parquet_path]
```

Set `EVAL_WORKERS` to control evaluation parallelism.

---

## Related benchmarks and methods

| Method / benchmark | Credit signal | State assumption | Extra models | CLI-native structure |
|--------------------|---------------|------------------|--------------|----------------------|
| GRPO / group-relative RL | Episode reward in batch | None | None | No |
| GiGPO | Observation-anchored normalization | Repeated observations | None | No |
| HGPO | Hierarchical state-group normalization | Repeated state groups | None | No |
| LLM-as-judge / process reward | Learned or prompted judge | None | Judge LLM | No |
| **A³ (this work)** | Episode + turn + tree advantages from shell AST | No repeated-state requirement | None | **Yes (bash AST signatures)** |
| **σ-Reveal (this work)** | N/A (inference-time observation) | Partial filesystem view | None | **Yes (workspace harness)** |

| Benchmark | Interaction | Verification | Task focus |
|-----------|-------------|--------------|------------|
| SWE-bench | patch + tests | unit tests | real GitHub issues |
| Terminal-Bench | terminal workflows | env checks | CLI side effects |
| **ShellOps (this work)** | shell in repo FS | stdout / file state | lookup, aggregate, edit, mixed |

---

## Acknowledgements

This repository builds on the code infrastructure of `verl` and `verl-agent`. We thank these projects for supporting open research in large-scale reinforcement learning and agentic training.

---

## Citation

```bibtex
@misc{su2026learningcliagentsstructured,
      title={Learning CLI Agents with Structured Action Credit under Selective Observation}, 
      author={Haoyang Su and Ying Wen},
      year={2026},
      eprint={2605.08013},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.08013}, 
}
```
