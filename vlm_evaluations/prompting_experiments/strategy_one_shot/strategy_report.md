# Prompting Strategy Report: One-Shot In-Context Learning

**Strategy Name**: `one_shot`  
**Model**: `Qwen/Qwen3-VL-2B-Instruct`  
**Execution Runtime**: Google Colab GPU (Nvidia Tesla T4)  
**Inference Time**: 128.29 seconds (~1.28s per item)  
**Overall Accuracy**: **56.00%** (56/100)  

---

## Performance by Template

| Query Template | Total Queries | Correct Predictions | Accuracy | vs. Zero-Shot Direct |
| :--- | :---: | :---: | :---: | :---: |
| **`attr_relation`** | 34 | 30 | **88.2%** | -5.9% |
| **`two_hop`** | 33 | 23 | **69.7%** | 0.0% (Tied) |
| **`between_attr`** | 33 | 3 | **9.1%** | -6.1% |
| **Overall** | **100** | **56** | **56.0%** | **-4.0%** |

---

## Strategy Analysis & Exemplar Bias Effect

1. **Exemplar Semantic Bias**:
   - Providing a single tabletop demonstration (`laptop`, `table`, `cup`, `notebook`) slightly degraded performance on diverse outdoor/relational scenes.
   - In multiple instances, the model predicted common indoor/household distractors (`lamp`, `candle`, `plate`, `shirt`) when the correct entity was outdoor (`window`, `picture`, `bread`).
2. **`two_hop` Robustness**:
   - Multi-hop traversal was preserved at 69.7% (identical to zero-shot), indicating that the structural demonstration did not hurt multi-step relation chaining.
