# Vision-Language Model Spatial Reasoning & Scene Graph Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model: Qwen3-VL-2B](https://img.shields.io/badge/VLM-Qwen3--VL--2B--Instruct-orange.svg)](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
[![Hardware: Tesla T4](https://img.shields.io/badge/Hardware-Tesla%20T4%20GPU-green.svg)](https://www.nvidia.com)

Empirical research benchmarking the spatial reasoning capabilities of Vision-Language Models (VLMs), specifically [**`Qwen/Qwen3-VL-2B-Instruct`**](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), comparing **Direct Image-to-Text Chain-of-Thought** against **Structured Scene Graph Reasoning**.

---

## 📊 Benchmark Scoreboard

Evaluated across **100 GQA-derived images and compositional queries** on Cloud GPUs (Tesla T4, FP16):

| Strategy / Paradigm | Input Modality | Accuracy (%) | Exact Match (%) | Latency / Query | Primary Bottleneck / Strength |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Direct QA on Ground Truth SGs** | Graph (Text) | **79.00%** | **77.00%** | **0.17s** | **High speed & precision**; 85.3% on 1-hop, 66.7% on 2-hop |
| **Zero-Shot CoT (1024 Tokens)** | Raw Image + Text | **51.00%** | **30.00%** | **40.23s** | **Token truncation**: 100% of errors hit 1024-token budget |
| **Direct QA on Baseline Model SGs** | Model Generated SG | **25.00%** | **17.00%** | **0.17s** | **Spatial omission**: Old prompt produced only 16.8% spatial edges |
| **Spatial SGG + Direct QA (New)** | Spatial Model SG | **Evaluated on Colab** | - | **0.20s** | **Rich directional triples**: Boosted spatial relations to **58.8% (718 triples)** |

---

## 🔬 Key Research Findings

1. **The CoT Token Budget Paradox**:
   - Small VLMs (2B parameters) spend considerable tokens verbalizing background features.
   - When capped at 1,024 tokens, **100% of CoT errors (49/49)** were caused by running out of tokens mid-thought.
   - For all completed chains, exact accuracy was **100%**, showing the model possesses reasoning capacity but suffers from verbosity.
2. **Perception-Reasoning Decoupling**:
   - Direct reasoning over structured scene graphs operates at **0.17s per query** (a **236x speedup** over direct image CoT) while achieving **79.0% accuracy** on ground truth graphs.
3. **The Spatial SGG Prompting Principle**:
   - Generic scene graph generation prompts produce part-whole relations (`wearing`, `has`, `on`), resulting in poor downstream QA accuracy (**25.0%**).
   - Prompting with **Few-Shot + Embedded Spatial CoT** enforces decoupled sub-components and directional predicates (`to the left of`, `to the right of`, `above`, `below`, `in front of`, `behind`), increasing directional triples by **+420%** (from 138 to 718 triples).

---

## 🗂️ Repository Structure

```
├── Images/                                    # 100 benchmark images (GQA Image 001 - 100)
├── Queries/                                   # 100 benchmark queries (attr_relation, spatial_conjunction, two_hop)
├── Small Scene Graph/                         # 100 ground truth local scene graphs
├── Large Scene Graph/                         # 100 ground truth full-scene graphs
├── benchmark.json                             # Master index of 100 benchmark samples
│
├── scene_graphs_100_images_spatial.json       # 100 generated scene graphs with spatial focus (718 directional edges)
├── generated_scene_graphs_100.json            # Baseline generated scene graphs (old prompt)
│
├── BENCHMARK_RESEARCH_REPORT.md               # Detailed academic & empirical research report
├── README.md                                  # Project overview and instructions
│
├── code/                                      # Analysis, building, and validation tools
│   ├── analyze_spatial_sgg_100.py             # Predicate & spatial distribution auditor
│   ├── build_spatial_qa_benchmark.py          # Assembles Direct QA evaluation dataset
│   ├── build_colab_spatial_eval_nb.py         # Syncs Colab notebook
│   ├── repair_and_finalize_sgg.py             # Validates and cleans JSON token streams
│   ├── analyze_benchmark_results.py           # Computes scoreboard statistics
│   └── validate_benchmark.py                  # Schema & dataset integrity checker
│
├── vlm_evaluations/
│   ├── notebooks/
│   │   ├── Qwen3_VL_Spatial_SG_Evaluation_Colab.ipynb # 1-Click Colab GPU evaluation notebook
│   │   └── Qwen3_VL_Prompting_Strategies_Colab.ipynb  # Multi-strategy prompting notebook
│   └── prompting_experiments/                 # Raw predictions and comparative reports
│
├── kaggle_spatial_sgg_100/                    # Kaggle kernel for 100-image spatial SGG generation
└── kaggle_spatial_qa_eval/                    # Kaggle kernel for downstream Direct QA evaluation
```

---

## 🚀 Quick Start: Running on Google Colab

Run the Direct QA benchmark on the new spatial scene graphs in Google Colab:
1. Open [`vlm_evaluations/notebooks/Qwen3_VL_Spatial_SG_Evaluation_Colab.ipynb`](vlm_evaluations/notebooks/Qwen3_VL_Spatial_SG_Evaluation_Colab.ipynb) in [Google Colaboratory](https://colab.research.google.com).
2. Set runtime accelerator to **T4 GPU** (**Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ **T4 GPU**).
3. Select **Runtime** $\rightarrow$ **Run all**.
4. The 100 queries will complete in **~20 seconds**, displaying the accuracy scoreboard and error breakdown.

---

## 📜 Citation & Credits
Based on the GQA visual reasoning dataset and Hugging Face `Qwen/Qwen3-VL-2B-Instruct`.
