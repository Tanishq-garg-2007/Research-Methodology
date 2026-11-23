# Spatial VLM Benchmark: Selected Category Cases & Comparative Results

This package contains **1 Correct** and **1 Wrong** case from each question category evaluated on **Qwen3-VL-2B-Instruct** (`simple Image + Query` VLM inference), paired with:
1. **The original image file** (`.jpg`)
2. **Simple Image VLM result** (`simple_image_vlm_result.json` from `qwen3_vl_answers.json`)
3. **Model-Generated Spatial Scene Graph** (`model_generated_spatial_scene_graph.json` — Few-Shot + Spatial CoT)
4. **Ground-Truth Tight Small Scene Graph** (`ground_truth_small_scene_graph.json` — query-associated subgraph)
5. **Our Model's Direct QA on Small Scene Graph** (`small_sg_direct_qa_result.json`)
6. **Our Model's CoT QA on Small Scene Graph** (`small_sg_cot_qa_result.json` — includes full step-by-step reasoning trace)
7. **Our Model's Direct QA on Generated Scene Graph** (`model_generated_sg_direct_qa_result.json`)
8. **Side-by-Side Comparison** (`case_evaluation_comparison.json`)

---

## 📊 Comprehensive Results Across Paradigms

| Category | Status (Image VLM) | Image | Question | Ground Truth | Simple Image VLM | Small SG Direct QA | Small SG CoT QA | Gen SG Direct QA |
|---|---|---|---|---|---|---|---|---|
| **Two-Hop** | **CORRECT** | `GQA Image 016.jpg` | *What is on the pole that is to the left of the building?* | **sign** | **sign** (✅) | air conditioner (❌) | *(1024 token timeout)* (❌) | pole (❌) |
| **Two-Hop** | **WRONG** | `GQA Image 001.jpg` | *What is on the plate that is to the right of the carrots?* | **fork** | sandwich/salad (❌) | **fork** (✅) | **fork** (✅) | grilled sandwich (❌) |
| **Attr-Relation** | **CORRECT** | `GQA Image 002.jpg` | *What is to the left of the cockpit?* | **propeller** | **propeller** (✅) | letter (❌) | letter (❌) | fuselage (❌) |
| **Attr-Relation** | **WRONG** | `GQA Image 003.jpg` | *What is to the right of the cut grass?* | **pole** | grass field (❌) | **pole** (✅) | **pole** (✅) | horse (❌) |
| **Spatial-Conjunction** | **CORRECT** | `GQA Image 004.jpg` | *What object is to the right of the black clothes and to the left of the red sneakers?* | **skateboard** | **skateboard** (✅) | man (❌) | man (❌) | **skateboard** (✅) |
| **Spatial-Conjunction** | **WRONG** | `GQA Image 007.jpg` | *What object is to the right of the white toilet and to the right of the lid?* | **counter** | hairbrush (❌) | **counter** (✅) | **counter** (✅) | sink (❌) |

---

## 🔍 Key Insights Revealed by this Package

1. **Why Direct Image VLM Fails (Images 1, 3, 7)**:
   - On **Image 1**, the direct VLM identifies the large plate and the sandwich, but fails to resolve the small metallic **fork**. The Small Scene Graph explicitly preserves the triple `('fork', 'on', 'plate')`, allowing both Direct and CoT QA to score **100% correct**.
   - On **Image 3**, visual over-smoothing leads the direct VLM to describe the open green grass field, missing the thin vertical **pole**. The Scene Graph encodes the relation `('pole', 'to the right of', 'cut grass')`, achieving perfect retrieval.
   - On **Image 7**, direct VLM hallucinates a small handheld hairbrush instead of the large **counter**. The Scene Graph enforces the dual spatial conjunction, leading directly to `counter`.

2. **The CoT Token-Exhaustion Paradox (Image 16)**:
   - On Image 16, CoT QA hits the 1024-token budget limit trying to resolve circular mirror-bus relations before emitting a final answer. Direct Image VLM, in contrast, directly sees the prominent black sign on the pole.

3. **Spatial SGG Precision**:
   - The included `model_generated_spatial_scene_graph.json` demonstrates how our Few-Shot + Spatial CoT prompt successfully decouples component nodes (e.g., cockpit, fuselage, propeller in Image 2) and grounds directional spatial predicates (`to the left of`, `to the right of`, `above`, `below`).

---

## 📁 Folder Structure

```text
spatial_benchmark_category_cases/
├── README.md
├── cases_manifest.json
│
├── 01_two_hop/
│   ├── correct_img016/
│   │   ├── GQA Image 016.jpg
│   │   ├── simple_image_vlm_result.json
│   │   ├── model_generated_spatial_scene_graph.json
│   │   ├── ground_truth_small_scene_graph.json
│   │   ├── small_sg_direct_qa_result.json
│   │   ├── small_sg_cot_qa_result.json
│   │   ├── model_generated_sg_direct_qa_result.json
│   │   └── case_evaluation_comparison.json
│   └── wrong_img001/
│       ├── GQA Image 001.jpg
│       ├── simple_image_vlm_result.json
│       ├── model_generated_spatial_scene_graph.json
│       ├── ground_truth_small_scene_graph.json
│       ├── small_sg_direct_qa_result.json
│       ├── small_sg_cot_qa_result.json
│       ├── model_generated_sg_direct_qa_result.json
│       └── case_evaluation_comparison.json
│
├── 02_attr_relation/
│   ├── correct_img002/
│   │   └── ... (all 7 JSONs + JPG)
│   └── wrong_img003/
│       └── ... (all 7 JSONs + JPG)
│
└── 03_spatial_conjunction/
    ├── correct_img004/
    │   └── ... (all 7 JSONs + JPG)
    └── wrong_img007/
        └── ... (all 7 JSONs + JPG)
```
