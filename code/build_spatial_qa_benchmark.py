import json, gzip, base64, os

os.makedirs('kaggle_spatial_qa_eval', exist_ok=True)

# 1. Load the original 100 query items from eval_gen_sg.py
with open('kaggle_gen_sg_eval/eval_gen_sg.py', 'r', encoding='utf-8') as f:
    text = f.read()

samples = None
for line in text.splitlines():
    if line.startswith('SAMPLES_B64 = '):
        b64 = line.split('"')[1]
        samples = json.loads(gzip.decompress(base64.b64decode(b64.encode('ascii'))).decode('utf-8'))
        break

if not samples:
    raise RuntimeError("Failed to load baseline samples from eval_gen_sg.py")

print(f"Loaded {len(samples)} baseline query items.")

# 2. Load the newly generated spatial scene graphs
with open('kaggle_spatial_sgg_100/output/scene_graphs_100_images_spatial.json', 'r', encoding='utf-8') as f:
    new_sgs_list = json.load(f)

new_sgs_by_img = {}
for item in new_sgs_list:
    img_num = item["image_number"]
    new_sgs_by_img[img_num] = item.get("scene_graph", {})

print(f"Loaded {len(new_sgs_by_img)} newly generated spatial scene graphs.")

# 3. Merge: replace objects and relations with the new spatial scene graph
merged_samples = []
for s in samples:
    img_num = s["image_number"]
    sg = new_sgs_by_img.get(img_num, {})
    
    raw_objs = sg.get("objects", [])
    raw_rels = sg.get("relations", [])
    
    # Normalize objects
    norm_objs = []
    for o in raw_objs:
        name = o.get("name", o.get("label", "unknown")).replace("_", " ")
        attrs = o.get("attributes", [])
        norm_objs.append({"label": name, "attributes": attrs})
        
    # Normalize relations
    norm_rels = []
    for r in raw_rels:
        sub = r.get("subject", "").replace("_", " ")
        rel = r.get("relation", "").replace("_", " ")
        obj = r.get("object", "").replace("_", " ")
        norm_rels.append({"subject": sub, "relation": rel, "object": obj})
        
    merged_item = {
        "sample_index": s["sample_index"],
        "image_number": img_num,
        "template": s["template"],
        "question": s["question"],
        "ground_truth": s["ground_truth"],
        "objects": norm_objs,
        "relations": norm_rels
    }
    merged_samples.append(merged_item)

print(f"Created {len(merged_samples)} merged evaluation samples.")

# Compress samples to base64 string
samples_json = json.dumps(merged_samples)
samples_b64 = base64.b64encode(gzip.compress(samples_json.encode('utf-8'))).decode('ascii')

# 4. Create the evaluation script for Kaggle
script_template = '''# =====================================================================
# Direct Benchmark Evaluation on New Spatial-Focused Scene Graphs
# Model: Qwen/Qwen3-VL-2B-Instruct
# Input: 100 Spatial Scene Graphs (Generated with Few-Shot + Spatial CoT)
# Strategy: Direct Question Answering (No CoT, max_tokens 32)
# Accelerator: Kaggle GPU (Tesla T4)
# =====================================================================
import os, sys, time, json, re, gzip, base64, subprocess

print("[INFO] Installing required dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "qwen-vl-utils", "transformers>=4.49.0", "accelerate"])

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required but not available.")
print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} (Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

model_id = "Qwen/Qwen3-VL-2B-Instruct"
print(f"[INFO] Loading model {model_id} in float16...")

from transformers import AutoProcessor, AutoModelForImageTextToText
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True
)
print("[INFO] Model loaded successfully.")
model.eval()

# Embedded test samples (100 items with new spatial scene graphs)
SAMPLES_B64 = "__SAMPLES_B64__"
SAMPLES = json.loads(gzip.decompress(base64.b64decode(SAMPLES_B64.encode('ascii'))).decode('utf-8'))
print(f"[INFO] Successfully loaded {len(SAMPLES)} benchmark items.")

def format_sg(item):
    objects = item.get("objects", [])
    relations = item.get("relations", [])
    obj_lines = []
    for obj in objects:
        lbl = obj.get("label", "unknown")
        attrs = obj.get("attributes", [])
        attr_str = f"attributes: [{', '.join(attrs)}]" if attrs else "no attributes"
        obj_lines.append(f"  - {lbl} ({attr_str})")

    rel_lines = []
    for rel in relations:
        s = rel.get("subject", "")
        r = rel.get("relation", "")
        o = rel.get("object", "")
        rel_lines.append(f'  - "{s}" {r} "{o}"')
        
    return len(objects), "\\n".join(obj_lines) if obj_lines else "  None", len(relations), "\\n".join(rel_lines) if rel_lines else "  None"

# DIRECT PROMPT (NO CHAIN-OF-THOUGHT)
def build_direct_prompt(item):
    num_obj, obj_str, num_rel, rel_str = format_sg(item)
    q = item.get("question", "")
    return (
        "You are given a scene graph representation of an image containing objects and their relationships, followed by a question.\\n\\n"
        f"Scene Graph:\\nObjects ({num_obj}):\\n{obj_str}\\n\\n"
        f"Relationships ({num_rel}):\\n{rel_str}\\n\\n"
        f"Question: {q}\\n\\n"
        "Instructions: Based strictly on the scene graph provided above, answer the question accurately. "
        "Provide only the direct concise answer (a few words or a single term) without explanation or reasoning.\\n"
        "Answer:"
    )

def extract_direct_answer(pred_text):
    clean = pred_text.strip()
    clean = re.sub(r"^(?:answer|the answer is|final answer):\\s*", "", clean, flags=re.IGNORECASE).strip()
    first_line = clean.split("\\n")[0].strip()
    first_line = re.sub(r"^(?:answer|the answer is|final answer):\\s*", "", first_line, flags=re.IGNORECASE).strip()
    return first_line.rstrip(".")

print("\\n" + "="*75)
print(f"STARTING EVALUATION ON NEW SPATIAL SCENE GRAPHS (ALL {len(SAMPLES)} ITEMS, MAX_TOKENS 32)")
print("="*75 + "\\n")

results = []
correct_count = 0
exact_count = 0
cat_stats = {
    "attr_relation": {"total": 0, "correct": 0, "exact": 0},
    "two_hop": {"total": 0, "correct": 0, "exact": 0},
    "spatial_conjunction": {"total": 0, "correct": 0, "exact": 0}
}

total_start_time = time.time()

for i, item in enumerate(SAMPLES, 1):
    gt = str(item.get("ground_truth", "")).strip().lower()
    q = item["question"]
    t = item["template"]
    cat_stats[t]["total"] += 1
    
    prompt = build_direct_prompt(item)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], return_tensors="pt").to("cuda")
    
    start_t = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            repetition_penalty=1.15
        )
    elapsed = time.time() - start_t
    
    gen_ids = out_ids[0][len(inputs["input_ids"][0]):]
    raw_pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
    gen_token_count = len(gen_ids)
    
    parsed = extract_direct_answer(raw_pred)
    clean_pred = parsed.lower().rstrip(".").strip()
    
    is_exact = (clean_pred == gt)
    is_correct = is_exact or (gt in clean_pred) or (clean_pred in gt and len(clean_pred) > 1)
    
    if is_correct:
        correct_count += 1
        cat_stats[t]["correct"] += 1
    if is_exact:
        exact_count += 1
        cat_stats[t]["exact"] += 1
        
    status = "EXACT" if is_exact else ("CORRECT" if is_correct else "WRONG")
    
    res_entry = {
        "sample_index": item["sample_index"],
        "image_number": item["image_number"],
        "template": t,
        "question": q,
        "ground_truth": gt,
        "tokens_generated": gen_token_count,
        "latency_sec": elapsed,
        "raw_output": raw_pred,
        "predicted_answer": parsed,
        "is_correct": is_correct,
        "is_exact": is_exact
    }
    results.append(res_entry)
    
    if i % 10 == 0 or i == len(SAMPLES) or not is_correct:
        print(f"[{i:03d}/{len(SAMPLES)}] Img {item['image_number']:03d} | {t[:12]:12} | {elapsed:.2f}s | {status:7} | Pred: '{parsed}' | GT: '{gt}'")

total_time = time.time() - total_start_time
acc = (correct_count / len(SAMPLES)) * 100
exact_acc = (exact_count / len(SAMPLES)) * 100

print("\\n" + "="*75)
print("FINAL BENCHMARK EVALUATION SUMMARY (NEW SPATIAL SCENE GRAPHS)")
print("="*75)
print(f"Total Samples: {len(SAMPLES)}")
print(f"Overall Accuracy: {acc:.2f}% ({correct_count}/{len(SAMPLES)})")
print(f"Exact Match Accuracy: {exact_acc:.2f}% ({exact_count}/{len(SAMPLES)})")
print(f"Total Inference Time: {total_time:.2f}s (Avg: {total_time/len(SAMPLES):.3f}s/query)")
print("\\nCategory Breakdown:")
for cat, st in cat_stats.items():
    tot = st["total"]
    c_acc = (st["correct"] / tot * 100) if tot > 0 else 0
    e_acc = (st["exact"] / tot * 100) if tot > 0 else 0
    print(f"  {cat:20}: {c_acc:6.2f}% Acc ({st['correct']}/{tot}) | Exact: {e_acc:6.2f}% ({st['exact']}/{tot})")
print("="*75)

final_payload = {
    "strategy": "Direct Prompting (Spatial Model-Generated Scene Graphs)",
    "total": len(SAMPLES),
    "correct": correct_count,
    "exact": exact_count,
    "accuracy": acc,
    "exact_accuracy": exact_acc,
    "total_time_seconds": total_time,
    "avg_time_per_sample": total_time / len(SAMPLES),
    "category_breakdown": {
        cat: {
            "total": st["total"],
            "correct": st["correct"],
            "exact": st["exact"],
            "accuracy": (st["correct"] / st["total"] * 100) if st["total"] > 0 else 0,
            "exact_accuracy": (st["exact"] / st["total"] * 100) if st["total"] > 0 else 0
        }
        for cat, st in cat_stats.items()
    },
    "results": results
}

with open("spatial_sg_benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(final_payload, f, indent=2)

print("[INFO] Results saved to spatial_sg_benchmark_results.json successfully.")
'''

full_script = script_template.replace('__SAMPLES_B64__', samples_b64)

with open('kaggle_spatial_qa_eval/eval_spatial_sg.py', 'w', encoding='utf-8') as f:
    f.write(full_script)

metadata = {
    "id": "tanishqgarg2007/qwen3-spatial-sg-eval",
    "title": "qwen3 spatial sg eval",
    "code_file": "eval_spatial_sg.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []
}

with open('kaggle_spatial_qa_eval/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("Kaggle package created successfully in kaggle_spatial_qa_eval/")
