# Research Methodology: Scene Graph Understanding Benchmark with VLMs

This repository contains the benchmark datasets, evaluation pipelines, and empirical findings for evaluating Vision-Language Models (VLMs) — specifically [**`Qwen/Qwen3-VL-2B-Instruct`**](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) — on structured scene graph comprehension.

---

## 1. Benchmark Overview

The benchmark evaluates the model's ability to reason over compact local scene graphs derived from the GQA dataset across 100 sample images (`GQA Image 001` through `GQA Image 100`):
- **Small Scene Graph** (`Small Scene Graph/`): Localized subgraphs containing anchor entities, core nodes, object attributes, normalized bounding boxes (`[x, y, width, height]`), and explicit relational edges.
- **Queries** (`Queries/`): Relational and compositional question-answer pairs categorized into three distinct templates:
  - **`attr_relation`** (34 items): Single-hop spatial relations with attribute constraints (e.g., *"What is to the left of the cockpit?"*).
  - **`two_hop`** (33 items): Sequential two-step relational traversals (e.g., *"What is on the plate that is to the right of the carrots?"*).
  - **`between_attr`** (33 items): Spatial betweenness queries requiring coordinate interval testing between two anchors (e.g., *"What is between the black clothes and the red sneakers?"*).

---

## 2. Overall Results Summary

All evaluations were executed on cloud GPU compute (`Nvidia Tesla T4`, `torch.bfloat16`).

### Scoreboard: Direct Baseline & Prompting Strategies

| Experiment / Strategy | Total Items | Accuracy | `attr_relation` (34) | `two_hop` (33) | `between_attr` (33) | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct Baseline (`zero_shot_direct`)** | 100 | **60.0%** | **94.1% (32/34)** | **69.7% (23/33)** | **15.2% (5/33)** | **83.2s** |
| **`one_shot` (Demonstration Learning)** | 100 | **56.0%** | 88.2% (30/34) | **69.7% (23/33)** | 9.1% (3/33) | 128.3s |
| **`zero_shot_cot` (Chain-of-Thought)** | 100 | **17.0%** | 32.4% (11/34) | 6.1% (2/33) | 12.1% (4/33) | 798.1s |

---

## 3. Key Findings

1. **Direct Single-Hop Precision (`attr_relation`: 94.1%)**:
   - The model reliably resolves relational triples when explicit edges exist in the scene graph.
2. **Two-Hop Traversal Dynamics (`two_hop`: 69.7%)**:
   - The primary error mode was **premature termination at Hop 1** (e.g., returning the intermediate anchor entity such as `plate` instead of continuing to the final target `fork`).
3. **The Betweenness Bottleneck (`between_attr`: 15.2%)**:
   - Because betweenness is not an explicit edge in the graph, it requires continuous 1D/2D bounding-box arithmetic ($\text{center}_x(A) < \text{center}_x(C) < \text{center}_x(B)$). Small VLMs struggle with raw floating-point interval comparisons without tool-augmented calculators.
4. **Prompting Strategy Trade-offs**:
   - **Zero-Shot Direct** yielded the highest accuracy (60.0%) and lowest latency.
   - **One-Shot** introduced slight domain bias toward tabletop/indoor entities from the exemplar.
   - **Zero-Shot CoT** suffered from generation length truncation when capped at 128 tokens; verbose reasoning ran out of tokens before emitting `Final Answer`.

---

## 4. Repository Structure

```
.
├── Large Scene Graph/                          # Full image scene graphs
├── Small Scene Graph/                          # Compact local scene graphs (100 JSONs)
├── Queries/                                    # Updated relational queries (100 JSONs)
├── benchmark.json                              # Consolidated dataset index
├── VALIDATION.md                               # Graph and query validation report
├── README.md                                   # Project documentation
│
└── vlm_evaluations/
    ├── direct_evaluation/                      # Direct VLM baseline evaluation
    │   ├── qwen3_vl_updated_results.json       # Complete predictions & ground truths
    │   ├── qwen3_vl_updated_evaluation_report.md# Comprehensive evaluation write-up
    │   ├── between_attr_analysis_report.md     # Root-cause analysis on betweenness
    │   ├── qwen3_vl_error_analysis_report.md   # Itemized analysis of all 40 error cases
    │   └── run_direct_eval.py                  # Self-contained GPU inference script
    │
    ├── prompting_experiments/                  # Dedicated strategy folders & comparison
    │   ├── strategy_zero_shot_direct/          # Zero-shot direct prompt, results & report
    │   ├── strategy_one_shot/                  # One-shot exemplar prompt, results & report
    │   ├── strategy_zero_shot_cot/             # Chain-of-thought prompt, results & report
    │   ├── results_zero_shot_direct.json       # Raw zero-shot direct outputs
    │   ├── results_one_shot.json               # Raw one-shot outputs
    │   ├── results_zero_shot_cot.json          # Raw zero-shot CoT outputs
    │   ├── strategies_summary.json             # Metrics comparison scoreboard
    │   └── comparative_prompting_analysis_report.md # Multi-strategy research report
    │
    └── notebooks/
        └── Qwen3_VL_Prompting_Strategies_Colab.ipynb # 1-click Colab GPU notebook
```

---

## 5. Quick Start: Reproducing Results on Colab

To reproduce all prompting strategies on Google Colab GPU:
1. Open [`vlm_evaluations/notebooks/Qwen3_VL_Prompting_Strategies_Colab.ipynb`](vlm_evaluations/notebooks/Qwen3_VL_Prompting_Strategies_Colab.ipynb) in [Google Colab](https://colab.research.google.com).
2. Set runtime accelerator to **T4 GPU** (**Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ **T4 GPU**).
3. Select **Runtime** $\rightarrow$ **Run all**.
