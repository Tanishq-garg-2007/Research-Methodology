# Prompting Strategy Report: Zero-Shot Direct

**Strategy Name**: `zero_shot_direct`  
**Model**: `Qwen/Qwen3-VL-2B-Instruct`  
**Execution Runtime**: Google Colab GPU (Nvidia Tesla T4)  
**Inference Time**: 83.2 seconds (~0.83s per item)  
**Overall Accuracy**: **60.00%** (60/100)  

---

## Performance by Template

| Query Template | Total Queries | Correct Predictions | Accuracy |
| :--- | :---: | :---: | :---: |
| **`attr_relation`** | 34 | 32 | **94.1%** |
| **`two_hop`** | 33 | 23 | **69.7%** |
| **`between_attr`** | 33 | 5 | **15.2%** |
| **Overall** | **100** | **60** | **60.0%** |

---

## Strategy Analysis & Key Strengths

1. **Top Overall Accuracy (60.0%)**:
   - Zero-Shot Direct proved to be the best performing strategy on this benchmark.
   - Constraining the generation strictly to the direct target entity (`max_new_tokens=24`) avoids hallucination, rambling, and semantic drift.
2. **Dominant on `attr_relation` (94.1%)**:
   - The model directly maps single-hop relational predicates with high precision.
3. **High Efficiency**:
   - Total runtime was only 83.2 seconds on T4 GPU, making it ~1.5x faster than One-Shot and ~9.6x faster than CoT.
