# Qwen/Qwen3-VL-2B-Instruct Evaluation Report (Updated Benchmark)

**Date**: November 21, 2026  
**Model**: `Qwen/Qwen3-VL-2B-Instruct` (Hugging Face)  
**Hardware Platform**: Kaggle GPU (`Nvidia Tesla T4`, 15.96 GB VRAM, `torch.bfloat16`)  
**Results JSON**: [`qwen3_vl_updated_results.json`](file:///c:/scene_understanding/qwen3_vl_updated_results.json)  
**Kaggle Kernel**: [`di5hagupta/qwen3-vl-updated-eval`](https://www.kaggle.com/code/di5hagupta/qwen3-vl-updated-eval)

---

## 1. Executive Summary

The evaluation of **Qwen/Qwen3-VL-2B-Instruct** on the updated benchmark suite from [Research-Methodology](https://github.com/Tanishq-garg-2007/Research-Methodology.git) has completed successfully on Kaggle Cloud GPU compute.

The updated benchmark replaced the 25 counting queries with relational and multi-hop queries, focusing evaluation on spatial-relational and multi-hop graph comprehension:
- **`attr_relation`**: 34 items
- **`two_hop`**: 33 items
- **`between_attr`**: 33 items
- **Total**: 100 items

---

## 2. Performance Comparison: v1 (Initial) vs v2 (Updated)

| Metric / Template | Initial Benchmark (v1) | Updated Benchmark (v2) | Delta |
| :--- | :---: | :---: | :---: |
| **`attr_relation`** | 96.0% (24/25) | **94.1% (32/34)** | -1.9% |
| **`two_hop`** | 80.0% (20/25) | **69.7% (23/33)** | -10.3% |
| **`counting`** | 28.0% (7/25) | *Removed in update* | N/A |
| **`between_attr`** | 12.0% (3/25) | **15.2% (5/33)** | +3.2% |
| **Overall Accuracy** | **54.0% (54/100)** | **60.0% (60/100)** | **+6.0%** |
| **Execution Time** | ~70 minutes (CPU) | **89.9 seconds (Nvidia T4 GPU)** | **~47x Speedup** |

---

## 3. Detailed Results Breakdown

### Template Performance Summary

| Template | Total Queries | Correct Predictions | Exact Matches | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **`attr_relation`** | 34 | 32 | 30 | **94.12%** |
| **`two_hop`** | 33 | 23 | 20 | **69.70%** |
| **`between_attr`** | 33 | 5 | 5 | **15.15%** |
| **Overall Total** | **100** | **60** | **55** | **60.00%** |

---

## 4. Qualitative Analysis & Sample Predictions

### 4.1. `attr_relation` (94.12% Accuracy — 32/34 Correct)
The model consistently and accurately resolves single-hop spatial relations with attribute constraints.
- **Image 002**:
  - *Question*: "What is to the left of the cockpit?"
  - *Prediction*: `propeller` | *Ground Truth*: `propeller` (Correct)
- **Image 003**:
  - *Question*: "What is to the right of the cut grass?"
  - *Prediction*: `pole` | *Ground Truth*: `pole` (Correct)
- **Image 025**:
  - *Question*: "What is to the right of the black shoes?"
  - *Prediction*: `animal` | *Ground Truth*: `animal` (Correct)
- **Errors**:
  - *Image 083*: "What is to the right of the horse?" $\rightarrow$ Predicted: `horse`, GT: `trash` (returned the subject itself).
  - *Image 088*: "What is to the right of the orange plate?" $\rightarrow$ Predicted: `cake`, GT: `knife`.

### 4.2. `two_hop` (69.70% Accuracy — 23/33 Correct)
The model navigates two sequential relations in the graph effectively.
- **Image 005**:
  - *Question*: "What is on the tree that is to the left of the bush?"
  - *Prediction*: `trunk` | *Ground Truth*: `trunk` (Correct)
- **Image 010**:
  - *Question*: "What is to the right of the skin that is to the right of the earphones?"
  - *Prediction*: `trees` | *Ground Truth*: `trees` (Correct)
- **Primary Failure Mode (Premature Termination at Hop 1)**:
  - *Image 001*: "What is on the plate that is to the right of the carrots?" $\rightarrow$ Predicted `plate`, GT: `fork`.
  - *Image 006*: "What is on the shelf that is to the right of the chair?" $\rightarrow$ Predicted `shelf`, GT: `container`.
  - *Analysis*: The model sometimes stops at the intermediate anchor entity (`plate`, `shelf`) identified in Hop 1 rather than completing Hop 2 to retrieve the object residing on it.

### 4.3. `between_attr` (15.15% Accuracy — 5/33 Correct)
- **Successful Cases**:
  - *Image 008*: "What is between the hand and the large sign?" $\rightarrow$ `window` (Correct)
  - *Image 020*: "What is between the chair and the tall plant?" $\rightarrow$ `window` (Correct)
- **Failure Analysis**:
  - *Image 004*: "What is between the black clothes and the red sneakers?" $\rightarrow$ Predicted `skateboard`, GT: `cap`.
  - *Image 007*: "What is between the blue comb and the lid?" $\rightarrow$ Predicted `counter`, GT: `toilet`.
  - *Image 009*: "What is between the frisbee and the white shirt?" $\rightarrow$ Predicted `hand`, GT: `man`.
  - *Root Cause*: The scene graph contains bounding boxes `[x, y, w, h]` for each object, but does not provide pre-computed explicit `between` edges. Spatial betweenness requires computing:
    $$\min(x_{A}, x_{B}) < x_{C} < \max(x_{A}, x_{B})$$
    Without chain-of-thought or explicit relational edge materialization, the VLM guesses salient contextual objects instead of performing continuous coordinate interpolation.

---

## 5. Artifacts and Generated Deliverables

| File / Resource | Location | Description |
| :--- | :--- | :--- |
| **Updated Results JSON** | [`qwen3_vl_updated_results.json`](file:///c:/scene_understanding/qwen3_vl_updated_results.json) | Complete predictions, ground truths, and exact match flags for all 100 images |
| **Initial Results JSON** | [`qwen3_vl_small_sg_results.json`](file:///c:/scene_understanding/qwen3_vl_small_sg_results.json) | Baseline evaluation results on the initial query set |
| **Between-Attr Diagnostic Report** | [`between_attr_analysis_report.md`](file:///c:/scene_understanding/between_attr_analysis_report.md) | In-depth root cause analysis and geometric breakdown of between_attr queries |
| **Evaluation Script** | [`run_eval.py`](file:///c:/scene_understanding/kaggle_qwen3_gpu_eval_v2/run_eval.py) | Self-contained GPU evaluation script with transformers source build and dynamic prompt generation |
| **Kaggle Kernel** | [`di5hagupta/qwen3-vl-updated-eval`](https://www.kaggle.com/code/di5hagupta/qwen3-vl-updated-eval) | Kaggle Cloud GPU execution run |
