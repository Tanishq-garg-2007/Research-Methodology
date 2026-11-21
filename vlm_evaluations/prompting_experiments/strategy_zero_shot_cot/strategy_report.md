# Prompting Strategy Report: Zero-Shot Chain-of-Thought (CoT)

**Strategy Name**: `zero_shot_cot`  
**Model**: `Qwen/Qwen3-VL-2B-Instruct`  
**Execution Runtime**: Google Colab GPU (Nvidia Tesla T4)  
**Inference Time**: 798.11 seconds (~7.98s per item)  
**Overall Accuracy**: **17.00%** (17/100)  

---

## Performance by Template

| Query Template | Total Queries | Correct Predictions | Accuracy | vs. Zero-Shot Direct |
| :--- | :---: | :---: | :---: | :---: |
| **`attr_relation`** | 34 | 11 | **32.4%** | -61.8% |
| **`two_hop`** | 33 | 2 | **6.1%** | -63.6% |
| **`between_attr`** | 33 | 4 | **12.1%** | -3.0% |
| **Overall** | **100** | **17** | **17.0%** | **-43.0%** |

---

## Critical Finding: Token Budget Truncation Bottleneck

1. **Why Accuracy Dropped to 17%**:
   - Detailed inspection of the raw outputs revealed that the model was performing genuine, correct intermediate reasoning (e.g. Identifying Step 1 anchors and Step 2 coordinate bounds).
   - However, because `max_new_tokens` was capped at 128, the model's verbose 3-step reasoning **consistently exhausted the token limit midway through Step 2 or Step 3** before reaching `Final Answer: <target>`.
   - The answer extraction parser was forced to fall back on incomplete trailing fragments (e.g. `'Step 3:'`, `'- The "grass" object is at [0.20, 0.54, 0.13, 0.11'`), registering false negatives.
2. **Computational Cost**:
   - CoT required 798.1 seconds (~13.3 minutes), nearly an order of magnitude slower than direct generation.
3. **Key Lesson**:
   - For small 2B parameter VLMs, multi-step textual reasoning must either have a large token generation budget ($\ge 384$ tokens) or use concise, structured scratchpads rather than free-form verbosity.
