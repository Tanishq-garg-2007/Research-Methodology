import os
import json
import gzip
import base64

def main():
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

    selected_cases = [
        {
            "category": "Two-Hop Spatial Traversal (`two_hop`)",
            "category_code": "two_hop",
            "verdict": "CORRECT",
            "image_number": 16,
            "filename": "GQA Image 016.jpg",
            "gqa_id": vlm_raw_results[16].get('gqa_image_id'),
            "analysis": "The query asks for the object on the pole to the left of the building. The Direct VLM on the image successfully focused on the bus-pole structure and identified the 'black sign'. In the Scene Graph Direct QA, the model confused the spatial ordering with an adjacent air conditioner, while CoT QA hit the 1024-token budget limit tracing cyclic mirror/bus attachments."
        },
        {
            "category": "Two-Hop Spatial Traversal (`two_hop`)",
            "category_code": "two_hop",
            "verdict": "WRONG",
            "image_number": 1,
            "filename": "GQA Image 001.jpg",
            "gqa_id": vlm_raw_results[1].get('gqa_image_id'),
            "analysis": "The query asks for what is on the plate to the right of the carrots. The Direct VLM identified the plate, sandwich, pickle, and salad, but missed the small metallic fork resting on the plate. However, on the Small Scene Graph, the explicit structured relation ('fork', 'on', 'plate') allowed both Direct QA and CoT QA to correctly answer 'fork'."
        },
        {
            "category": "Attribute & Directional Relation (`attr_relation`)",
            "category_code": "attr_relation",
            "verdict": "CORRECT",
            "image_number": 2,
            "filename": "GQA Image 002.jpg",
            "analysis": "The query asks for what is to the left of the cockpit. The Direct VLM correctly recognized the propeller on the aircraft wing. In our generated spatial scene graph, Few-Shot + Spatial CoT decoupled the fuselage, wing, and propeller with explicit directional relations, solving the prior failure where baseline scene graphs lumped all components into a generic aircraft node."
        },
        {
            "category": "Attribute & Directional Relation (`attr_relation`)",
            "category_code": "attr_relation",
            "verdict": "WRONG",
            "image_number": 3,
            "filename": "GQA Image 003.jpg",
            "analysis": "The query asks for what is to the right of the cut grass. The Direct VLM hallucinated a broad open grass lawn. In contrast, the Small Scene Graph directly encoded ('pole', 'to the right of', 'cut grass'), enabling both Direct QA and CoT QA to score 100% correct by answering 'pole'."
        },
        {
            "category": "Multi-Constraint Spatial Conjunction (`spatial_conjunction`)",
            "category_code": "spatial_conjunction",
            "verdict": "CORRECT",
            "image_number": 4,
            "filename": "GQA Image 004.jpg",
            "analysis": "The query requires bounding an object between two items: to the right of black clothes AND to the left of red sneakers. Direct Image VLM successfully bounded the geometry and answered 'skateboard'. Direct QA on our Model-Generated Scene Graph also achieved 100% accuracy with 'skateboard'."
        },
        {
            "category": "Multi-Constraint Spatial Conjunction (`spatial_conjunction`)",
            "category_code": "spatial_conjunction",
            "verdict": "WRONG",
            "image_number": 7,
            "filename": "GQA Image 007.jpg",
            "analysis": "The query asks for what object is to the right of the white toilet AND to the right of the lid. Direct Image VLM hallucinated a 'blue and black hairbrush'. However, on the Small Scene Graph, both Direct QA and CoT QA accurately resolved the dual right-hand spatial constraints to identify 'counter'."
        }
    ]

    md_lines = []
    md_lines.append("# 📊 Comprehensive Spatial VLM Benchmark Results")
    md_lines.append("")
    md_lines.append("This document aggregates all data from the benchmark case package across **3 question categories** (`two_hop`, `attr_relation`, `spatial_conjunction`), featuring **1 Correct** and **1 Wrong** case each.")
    md_lines.append("For every case, this file contains:")
    md_lines.append("1. **Image Metadata & Local Path**")
    md_lines.append("2. **Benchmark Query & Ground Truth Answer**")
    md_lines.append("3. **Direct VLM Output** (`Image + Query` from `qwen3_vl_answers.json`)")
    md_lines.append("4. **Our Model's Direct QA on Small Scene Graph**")
    md_lines.append("5. **Our Model's Chain-of-Thought (CoT) QA on Small Scene Graph** (including step-by-step reasoning trace)")
    md_lines.append("6. **Our Model's Direct QA on Generated Scene Graph**")
    md_lines.append("7. **Our Model's Generated Spatial Scene Graph** (Few-Shot + Spatial CoT)")
    md_lines.append("8. **Ground-Truth Tight Small Scene Graph** (with bounding boxes & attributes)")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 🏆 Master Comparison Table")
    md_lines.append("")
    md_lines.append("| Case # | Category | Status (Image VLM) | Image File | Question | Ground Truth | Simple Image VLM | Small SG Direct QA | Small SG CoT QA | Model Gen-SG Direct QA |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for idx, c in enumerate(selected_cases, 1):
        num = c['image_number']
        vlm_ans = vlm_raw_results[num].get('answer')
        # clean snippet
        v_snip = vlm_ans.replace("\n", " ")[:30] + "..." if len(vlm_ans) > 30 else vlm_ans
        d_pred = str(direct_qa_sg[num].get('predicted_answer', '')).replace('`', "'").replace('\n', ' ')
        c_pred = str(cot_qa_sg[num].get('predicted_answer', '')).replace('`', "'").replace('\n', ' ')
        if len(c_pred) > 40:
            c_pred = c_pred[:40] + "..."
        g_pred = str(direct_qa_gen_sg[num].get('predicted_answer', '')).replace('`', "'").replace('\n', ' ')
        q = vlm_raw_results[num].get('question')
        gt = vlm_raw_results[num].get('reference_answer')
        
        v_mark = "✅" if c['verdict'] == "CORRECT" else "❌"
        d_mark = "✅" if direct_qa_sg[num].get('is_correct') else "❌"
        c_mark = "✅" if cot_qa_sg[num].get('is_correct') else "❌"
        g_mark = "✅" if direct_qa_gen_sg[num].get('is_correct') else "❌"
        
        md_lines.append(f"| {idx} | {c['category_code']} | **{c['verdict']}** | `{c['filename']}` | *{q}* | **{gt}** | {v_mark} {v_snip} | {d_mark} {d_pred} | {c_mark} {c_pred} | {g_mark} {g_pred} |")
    
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 🖼️ Images Used in this Benchmark")
    md_lines.append("")
    md_lines.append("Here is the isolated list of all 6 images used, categorized by test scenario:")
    md_lines.append("")
    for idx, c in enumerate(selected_cases, 1):
        num = c['image_number']
        gqa_id = vlm_raw_results[num].get('gqa_image_id')
        md_lines.append(f"### {idx}. {c['filename']} (Image #{num})")
        md_lines.append(f"- **Scenario**: Category `{c['category_code']}` — **{c['verdict']}** in Direct Image VLM")
        md_lines.append(f"- **GQA Image ID**: `{gqa_id}`")
        md_lines.append(f"- **Workspace Path**: `Images/Images/{c['filename']}`")
        md_lines.append(f"- **Query**: *\"{vlm_raw_results[num].get('question')}\"*")
        md_lines.append(f"- **Target Answer**: **`{vlm_raw_results[num].get('reference_answer')}`**")
        md_lines.append("")

    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 📑 Detailed Case-by-Case Breakdown")
    md_lines.append("")

    for idx, c in enumerate(selected_cases, 1):
        num = c['image_number']
        v_res = vlm_raw_results[num]
        m_sgg = model_sgg[num]
        g_sgg = gt_items[num]
        d_res = direct_qa_sg[num]
        c_res = cot_qa_sg[num]
        g_res = direct_qa_gen_sg[num]

        gqa_id = v_res.get('gqa_image_id')
        md_lines.append(f"### Case {idx}: {c['category']} — [{c['verdict']}]")
        md_lines.append(f"- **Image**: `{c['filename']}` (Image #{num}, GQA ID: `{gqa_id}`)")
        md_lines.append(f"- **Question**: *\"{v_res.get('question')}\"*")
        md_lines.append(f"- **Ground Truth Answer**: **`{v_res.get('reference_answer')}`**")
        md_lines.append(f"- **Analysis**: {c['analysis']}")
        md_lines.append("")
        
        md_lines.append("#### 1. Simple Image + Query VLM Result (`qwen3_vl_answers.json`)")
        md_lines.append("```text")
        md_lines.append(f"Model: Qwen/Qwen3-VL-2B-Instruct")
        md_lines.append(f"Prompt: Image + Question directly")
        md_lines.append(f"Output:\n{v_res.get('answer')}")
        md_lines.append(f"Verdict: {c['verdict']}")
        md_lines.append("```")
        md_lines.append("")

        md_lines.append("#### 2. Our Model's Direct QA on Small Scene Graph (`kaggle_direct_results`)")
        md_lines.append("```json")
        md_lines.append(json.dumps({
            "model": "Qwen/Qwen3-VL-2B-Instruct",
            "prompt_type": "Direct QA on Scene Graph (No Bounding Boxes)",
            "predicted_answer": d_res.get('predicted_answer'),
            "ground_truth": d_res.get('ground_truth'),
            "is_correct": d_res.get('is_correct'),
            "latency_sec": d_res.get('latency_sec'),
            "tokens_generated": d_res.get('tokens_generated')
        }, indent=2))
        md_lines.append("```")
        md_lines.append("")

        md_lines.append("#### 3. Our Model's Chain-of-Thought (CoT) QA on Small Scene Graph (`kaggle_full_results`)")
        md_lines.append("```text")
        md_lines.append(f"Predicted Answer: {c_res.get('predicted_answer')} (Correct: {c_res.get('is_correct')})")
        md_lines.append(f"Latency: {c_res.get('latency_sec'):.2f}s | Tokens Generated: {c_res.get('tokens_generated')}")
        md_lines.append("Step-by-Step CoT Graph Traversal Reasoning:")
        md_lines.append(c_res.get('raw_output'))
        md_lines.append("```")
        md_lines.append("")

        md_lines.append("#### 4. Our Model's Direct QA on Model-Generated Scene Graph (`kaggle_gen_sg_results`)")
        md_lines.append("```json")
        md_lines.append(json.dumps({
            "model": "Qwen/Qwen3-VL-2B-Instruct",
            "prompt_type": "Direct QA on Generated Scene Graph",
            "predicted_answer": g_res.get('predicted_answer'),
            "ground_truth": g_res.get('ground_truth'),
            "is_correct": g_res.get('is_correct'),
            "latency_sec": g_res.get('latency_sec'),
            "tokens_generated": g_res.get('tokens_generated')
        }, indent=2))
        md_lines.append("```")
        md_lines.append("")

        md_lines.append("#### 5. Model-Generated Spatial Scene Graph (Our Model: Few-Shot + Spatial CoT)")
        md_lines.append(f"- **Summary**: {m_sgg.get('objects_count')} Objects, {m_sgg.get('relations_count')} Relations ({m_sgg.get('directional_relations_count')} Directional Spatial Triples)")
        md_lines.append("```json")
        md_lines.append(json.dumps(m_sgg.get('scene_graph', {}), indent=2))
        md_lines.append("```")
        md_lines.append("")

        md_lines.append("#### 6. Ground-Truth Tight Small Scene Graph (Query Reference)")
        md_lines.append(f"- **Summary**: Anchors: `{g_sgg.get('anchors')}` | Nodes: `{g_sgg.get('nodes')}` | {len(g_sgg.get('objects', []))} Objects, {len(g_sgg.get('relations', []))} Relations")
        md_lines.append("```json")
        md_lines.append(json.dumps({
            "anchors": g_sgg.get('anchors'),
            "nodes": g_sgg.get('nodes'),
            "objects": g_sgg.get('objects'),
            "relations": g_sgg.get('relations')
        }, indent=2))
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    full_text = "\n".join(md_lines)
    
    # Save to single.md
    with open("single.md", "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"[SUCCESS] Wrote single.md ({len(full_text)/1024:.1f} KB)")
    
    # Also save as single.mp in case user or tools look for exact extension
    with open("single.mp", "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"[SUCCESS] Wrote single.mp ({len(full_text)/1024:.1f} KB)")
    
    # Also mirror to Google Drive
    drive_dest = "G:/My Drive/RM"
    if os.path.exists(drive_dest):
        import shutil
        shutil.copy2("single.md", os.path.join(drive_dest, "single.md"))
        shutil.copy2("single.mp", os.path.join(drive_dest, "single.mp"))
        print(f"[SUCCESS] Mirrored single.md and single.mp to {drive_dest}")

if __name__ == "__main__":
    main()
