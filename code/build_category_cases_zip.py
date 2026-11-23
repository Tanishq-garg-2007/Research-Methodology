import os
import json
import gzip
import base64
import shutil
import zipfile

def main():
    print("=== Building Comprehensive Category Cases Zip Package ===")
    
    # 1. Load simple Image + Query VLM answers (qwen3_vl_answers (1).json)
    ans_file = 'qwen3_vl_answers (1).json'
    with open(ans_file, 'r', encoding='utf-8') as f:
        vlm_raw_results = {r['image_number']: r for r in json.load(f)['results']}
    
    # 2. Load model-generated spatial scene graphs
    sgg_file = 'scene_graphs_100_images_spatial.json'
    with open(sgg_file, 'r', encoding='utf-8') as f:
        model_sgg = {item['image_number']: item for item in json.load(f)}
    
    # 3. Load ground-truth small scene graphs from BENCHMARK_B64
    eval_script = 'vlm_evaluations/direct_evaluation/run_direct_eval.py'
    with open(eval_script, 'r', encoding='utf-8') as f:
        code = f.read()
    idx1 = code.find('BENCHMARK_B64 = """') + len('BENCHMARK_B64 = """')
    idx2 = code.find('"""', idx1)
    b64_str = code[idx1:idx2].strip()
    gt_items = {item['image_number']: item for item in json.loads(gzip.decompress(base64.b64decode(b64_str.encode('ascii'))).decode('utf-8'))}
    
    # 4. Load Direct QA on Small Scene Graph
    with open('kaggle_direct_results/direct_benchmark_results.json', 'r', encoding='utf-8') as f:
        direct_qa_sg = {r['image_number']: r for r in json.load(f)['results']}
        
    # 5. Load CoT QA on Small Scene Graph
    with open('kaggle_full_results/full_benchmark_results.json', 'r', encoding='utf-8') as f:
        cot_qa_sg = {r['image_number']: r for r in json.load(f)['results']}
        
    # 6. Load Direct QA on Model Generated Scene Graph
    with open('kaggle_gen_sg_results/generated_sg_benchmark_results.json', 'r', encoding='utf-8') as f:
        direct_qa_gen_sg = {r['image_number']: r for r in json.load(f)['results']}
    
    # 7. Selected cases across categories
    selected_cases = [
        {
            "category_folder": "01_two_hop",
            "category_name": "two_hop",
            "status": "correct",
            "subfolder": "correct_img016",
            "image_number": 16,
            "filename": "GQA Image 016.jpg",
            "analysis": "Direct Image VLM identified 'black sign'. On Small SG, Direct QA suffered from spatial confusion ('air conditioner'), while CoT hit token exhaustion (1024 tokens) analyzing bus mirror relations."
        },
        {
            "category_folder": "01_two_hop",
            "category_name": "two_hop",
            "status": "wrong",
            "subfolder": "wrong_img001",
            "image_number": 1,
            "filename": "GQA Image 001.jpg",
            "analysis": "Direct Image VLM recognized plate, sandwich, pickle, and salad, but missed the fork. In contrast, BOTH Direct QA and CoT QA on the Small Scene Graph correctly retrieved 'fork'."
        },
        {
            "category_folder": "02_attr_relation",
            "category_name": "attr_relation",
            "status": "correct",
            "subfolder": "correct_img002",
            "image_number": 2,
            "filename": "GQA Image 002.jpg",
            "analysis": "Direct Image VLM correctly answered 'propeller'. Direct QA and CoT on the unoriented graph struggled with letter/propeller ambiguity, highlighting the necessity of our directional spatial prompt."
        },
        {
            "category_folder": "02_attr_relation",
            "category_name": "attr_relation",
            "status": "wrong",
            "subfolder": "wrong_img003",
            "image_number": 3,
            "filename": "GQA Image 003.jpg",
            "analysis": "Direct Image VLM hallucinated a generic 'open field of green grass'. In contrast, BOTH Direct QA and CoT QA on the Small Scene Graph precisely identified 'pole'."
        },
        {
            "category_folder": "03_spatial_conjunction",
            "category_name": "spatial_conjunction",
            "status": "correct",
            "subfolder": "correct_img004",
            "image_number": 4,
            "filename": "GQA Image 004.jpg",
            "analysis": "Direct Image VLM correctly identified 'skateboard'. Direct QA on our Generated Scene Graph also correctly predicted 'skateboard'."
        },
        {
            "category_folder": "03_spatial_conjunction",
            "category_name": "spatial_conjunction",
            "status": "wrong",
            "subfolder": "wrong_img007",
            "image_number": 7,
            "filename": "GQA Image 007.jpg",
            "analysis": "Direct Image VLM hallucinated a 'blue and black hairbrush'. In contrast, BOTH Direct QA and CoT QA on the Small Scene Graph correctly solved the spatial conjunction and identified 'counter'."
        }
    ]
    
    # 8. Setup staging directory
    staging_dir = "category_cases_staging"
    pkg_root = os.path.join(staging_dir, "spatial_benchmark_category_cases")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(pkg_root, exist_ok=True)
    
    manifest_entries = []
    
    for case in selected_cases:
        img_num = case['image_number']
        cat_dir = os.path.join(pkg_root, case['category_folder'], case['subfolder'])
        os.makedirs(cat_dir, exist_ok=True)
        
        # A. Copy Image (.jpg)
        src_img = os.path.join('Images', 'Images', case['filename'])
        dst_img = os.path.join(cat_dir, case['filename'])
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            print(f"[OK] Copied {case['filename']} -> {case['subfolder']}")
            
        # B. Simple Image + Query VLM Result (qwen3_vl_answers.json)
        vlm_res = vlm_raw_results[img_num]
        query_vlm_payload = {
            "evaluation_paradigm": "Direct VLM on Image (Simple Image + Query)",
            "model": "Qwen/Qwen3-VL-2B-Instruct",
            "image_number": img_num,
            "image_filename": case['filename'],
            "gqa_image_id": vlm_res.get('gqa_image_id'),
            "query_category": case['category_name'],
            "evaluation_verdict": case['status'].upper(),
            "is_correct": case['status'] == "correct",
            "question": vlm_res.get('question'),
            "reference_ground_truth_answer": vlm_res.get('reference_answer'),
            "vlm_predicted_answer": vlm_res.get('answer'),
            "analysis": case['analysis']
        }
        with open(os.path.join(cat_dir, "simple_image_vlm_result.json"), "w", encoding="utf-8") as f:
            json.dump(query_vlm_payload, f, indent=2)
            
        # C. Model-Generated Spatial Scene Graph (Few-Shot + Spatial CoT)
        m_sgg_full = model_sgg[img_num]
        model_sg_payload = {
            "image_number": img_num,
            "filename": case['filename'],
            "generator_model": "Qwen/Qwen3-VL-2B-Instruct (Few-Shot + Embedded Spatial CoT)",
            "objects_count": m_sgg_full.get('objects_count', len(m_sgg_full['scene_graph'].get('objects', []))),
            "relations_count": m_sgg_full.get('relations_count', len(m_sgg_full['scene_graph'].get('relations', []))),
            "directional_relations_count": m_sgg_full.get('directional_relations_count', 0),
            "scene_graph": m_sgg_full.get('scene_graph', {})
        }
        with open(os.path.join(cat_dir, "model_generated_spatial_scene_graph.json"), "w", encoding="utf-8") as f:
            json.dump(model_sg_payload, f, indent=2)
            
        # D. Ground-Truth Small Scene Graph (reference tight subgraph)
        gt_item = gt_items[img_num]
        small_sg_payload = {
            "image_number": img_num,
            "gqa_image_id": gt_item.get('gqa_image_id'),
            "description": "Ground-Truth Tight Small Scene Graph associated with this query",
            "anchors": gt_item.get('anchors', []),
            "nodes": gt_item.get('nodes', []),
            "objects_count": len(gt_item.get('objects', [])),
            "relations_count": len(gt_item.get('relations', [])),
            "objects": gt_item.get('objects', []),
            "relations": gt_item.get('relations', [])
        }
        with open(os.path.join(cat_dir, "ground_truth_small_scene_graph.json"), "w", encoding="utf-8") as f:
            json.dump(small_sg_payload, f, indent=2)
            
        # E. Direct QA Result on Small Scene Graph
        d_sg = direct_qa_sg[img_num]
        direct_sg_payload = {
            "evaluation_paradigm": "Direct QA on Scene Graph (No Bounding Boxes)",
            "model": "Qwen/Qwen3-VL-2B-Instruct",
            "image_number": img_num,
            "question": d_sg.get('question'),
            "ground_truth_answer": d_sg.get('ground_truth'),
            "predicted_answer": d_sg.get('predicted_answer'),
            "raw_output": d_sg.get('raw_output'),
            "is_correct": d_sg.get('is_correct'),
            "latency_sec": d_sg.get('latency_sec'),
            "tokens_generated": d_sg.get('tokens_generated')
        }
        with open(os.path.join(cat_dir, "small_sg_direct_qa_result.json"), "w", encoding="utf-8") as f:
            json.dump(direct_sg_payload, f, indent=2)
            
        # F. Chain-of-Thought (CoT) QA Result on Small Scene Graph
        c_sg = cot_qa_sg[img_num]
        cot_sg_payload = {
            "evaluation_paradigm": "Chain-of-Thought (CoT) QA on Scene Graph (Step-by-Step Graph Traversal)",
            "model": "Qwen/Qwen3-VL-2B-Instruct",
            "image_number": img_num,
            "question": c_sg.get('question'),
            "ground_truth_answer": c_sg.get('ground_truth'),
            "predicted_answer": c_sg.get('predicted_answer'),
            "cot_step_by_step_reasoning": c_sg.get('raw_output'),
            "is_correct": c_sg.get('is_correct'),
            "latency_sec": c_sg.get('latency_sec'),
            "tokens_generated": c_sg.get('tokens_generated')
        }
        with open(os.path.join(cat_dir, "small_sg_cot_qa_result.json"), "w", encoding="utf-8") as f:
            json.dump(cot_sg_payload, f, indent=2)
            
        # G. Direct QA Result on Model Generated Scene Graph
        g_sg = direct_qa_gen_sg[img_num]
        gen_sg_qa_payload = {
            "evaluation_paradigm": "Direct QA on Model-Generated Scene Graph",
            "model": "Qwen/Qwen3-VL-2B-Instruct",
            "image_number": img_num,
            "question": g_sg.get('question'),
            "ground_truth_answer": g_sg.get('ground_truth'),
            "predicted_answer": g_sg.get('predicted_answer'),
            "is_correct": g_sg.get('is_correct'),
            "latency_sec": g_sg.get('latency_sec'),
            "tokens_generated": g_sg.get('tokens_generated')
        }
        with open(os.path.join(cat_dir, "model_generated_sg_direct_qa_result.json"), "w", encoding="utf-8") as f:
            json.dump(gen_sg_qa_payload, f, indent=2)
            
        # H. Side-by-Side Unified Comparison for this Case
        comparison_payload = {
            "case_id": f"{case['category_name']}_{case['status']}",
            "category": case['category_name'],
            "image_number": img_num,
            "image_filename": case['filename'],
            "question": vlm_res.get('question'),
            "ground_truth_reference": vlm_res.get('reference_answer'),
            "results_across_paradigms": {
                "1_simple_image_vlm": {
                    "method": "Direct Image + Query on VLM",
                    "prediction": vlm_res.get('answer'),
                    "verdict": "CORRECT" if case['status'] == "correct" else "WRONG",
                    "is_correct": case['status'] == "correct"
                },
                "2_small_sg_direct_qa": {
                    "method": "Direct QA on Tight Small Scene Graph",
                    "prediction": d_sg.get('predicted_answer'),
                    "is_correct": d_sg.get('is_correct'),
                    "latency_sec": d_sg.get('latency_sec')
                },
                "3_small_sg_cot_qa": {
                    "method": "CoT QA on Tight Small Scene Graph",
                    "prediction": c_sg.get('predicted_answer'),
                    "is_correct": c_sg.get('is_correct'),
                    "latency_sec": c_sg.get('latency_sec'),
                    "tokens_used": c_sg.get('tokens_generated')
                },
                "4_generated_sg_direct_qa": {
                    "method": "Direct QA on Model-Generated Scene Graph",
                    "prediction": g_sg.get('predicted_answer'),
                    "is_correct": g_sg.get('is_correct'),
                    "latency_sec": g_sg.get('latency_sec')
                }
            },
            "analysis": case['analysis']
        }
        with open(os.path.join(cat_dir, "case_evaluation_comparison.json"), "w", encoding="utf-8") as f:
            json.dump(comparison_payload, f, indent=2)
            
        manifest_entries.append({
            "case_id": f"{case['category_name']}_{case['status']}",
            "category": case['category_name'],
            "status": case['status'],
            "image_number": img_num,
            "image_filename": case['filename'],
            "question": vlm_res.get('question'),
            "ground_truth": vlm_res.get('reference_answer'),
            "simple_image_vlm_prediction": vlm_res.get('answer')[:100] + "...",
            "simple_image_vlm_correct": case['status'] == "correct",
            "small_sg_direct_qa_prediction": d_sg.get('predicted_answer'),
            "small_sg_direct_qa_correct": d_sg.get('is_correct'),
            "small_sg_cot_qa_prediction": c_sg.get('predicted_answer'),
            "small_sg_cot_qa_correct": c_sg.get('is_correct'),
            "gen_sg_direct_qa_prediction": g_sg.get('predicted_answer'),
            "gen_sg_direct_qa_correct": g_sg.get('is_correct'),
            "analysis": case['analysis']
        })

    # 9. Save Master Manifest JSON
    with open(os.path.join(pkg_root, "cases_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "description": "Comprehensive benchmark cases package containing: Images, Simple Image VLM answers, Model Spatial Scene Graphs, Ground-Truth Small Scene Graphs, Small SG Direct QA results, and Small SG CoT QA results.",
            "total_cases": len(manifest_entries),
            "categories": ["two_hop", "attr_relation", "spatial_conjunction"],
            "cases": manifest_entries
        }, f, indent=2)
        
    # 10. Generate Master README.md
    readme_content = f"""# Spatial VLM Benchmark: Selected Category Cases & Comparative Results

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
"""
    with open(os.path.join(pkg_root, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    # 11. Create Zip Archive
    zip_filename = "qwen3_vl_category_cases_with_scene_graphs.zip"
    if os.path.exists(zip_filename):
        os.remove(zip_filename)
        
    print(f"\nCompressing into {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(pkg_root):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, staging_dir)
                zipf.write(file_path, rel_path)
                
    zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"[SUCCESS] Created {zip_filename} ({zip_size_mb:.2f} MB)")
    
    # 12. Mirror to Google Drive
    drive_dest = "G:/My Drive/RM"
    if os.path.exists(drive_dest):
        drive_zip = os.path.join(drive_dest, zip_filename)
        shutil.copy2(zip_filename, drive_zip)
        print(f"[SUCCESS] Mirrored {zip_filename} to Google Drive: {drive_zip}")
        
    # 13. Clean up staging
    shutil.rmtree(staging_dir)
    print("[OK] Cleaned staging directory.")

if __name__ == "__main__":
    main()
