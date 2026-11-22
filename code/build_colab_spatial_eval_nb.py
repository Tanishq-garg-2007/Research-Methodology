import json, gzip, base64, os, shutil

# 1. Load merged samples (100 benchmark items with new spatial scene graphs)
with open('kaggle_spatial_qa_eval/eval_spatial_sg.py', 'r', encoding='utf-8') as f:
    text = f.read()

for line in text.splitlines():
    if line.startswith('SAMPLES_B64 = '):
        samples_b64 = line.split('"')[1]
        break

# 2. Build Colab Notebook structure
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "name": "Qwen3_VL_Spatial_SG_Evaluation_Colab.ipynb",
            "provenance": [],
            "gpuType": "T4"
        },
        "accelerator": "GPU",
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Direct QA Evaluation on Model-Generated Spatial Scene Graphs\n",
                "**Model**: `Qwen/Qwen3-VL-2B-Instruct`  \n",
                "**Scene Graphs**: 100 Full-Image Scene Graphs Generated with **Few-Shot + Embedded Spatial CoT**  \n",
                "**Strategy**: Direct Prompting (No CoT, max_new_tokens=32)  \n",
                "**Hardware**: Google Colab Free Tier (Tesla T4 GPU, ~20s total inference)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 1. Install dependencies\n",
                "!pip install -q qwen-vl-utils \"transformers>=4.49.0\" accelerate pillow pandas"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 2. Check GPU & Load Qwen3-VL-2B-Instruct\n",
                "import os, sys, time, json, re, gzip, base64, torch\n",
                "from transformers import AutoProcessor, AutoModelForImageTextToText\n",
                "\n",
                "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
                "print(f\"[INFO] Hardware: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'} | Device: {device}\")\n",
                "if device != \"cuda\":\n",
                "    print(\"[WARN] Please enable GPU: Runtime -> Change runtime type -> T4 GPU\")\n",
                "\n",
                "MODEL_ID = \"Qwen/Qwen3-VL-2B-Instruct\"\n",
                "print(f\"[INFO] Loading {MODEL_ID} in float16...\")\n",
                "processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)\n",
                "model = AutoModelForImageTextToText.from_pretrained(\n",
                "    MODEL_ID,\n",
                "    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,\n",
                "    device_map=\"auto\",\n",
                "    trust_remote_code=True\n",
                ")\n",
                "model.eval()\n",
                "print(\"[INFO] Model loaded successfully!\")"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                f"# 3. Load Embedded Benchmark Data (100 Items with Spatial Scene Graphs)\n",
                f'SAMPLES_B64 = "{samples_b64}"\n',
                "SAMPLES = json.loads(gzip.decompress(base64.b64decode(SAMPLES_B64.encode('ascii'))).decode('utf-8'))\n",
                "print(f\"[INFO] Successfully loaded {len(SAMPLES)} benchmark items.\")\n",
                "\n",
                "# Print sample preview\n",
                "sample_preview = SAMPLES[1]  # Image 002 (Airplane)\n",
                "print(f\"Sample Preview (Image {sample_preview['image_number']}):\")\n",
                "print(f\"  Question: {sample_preview['question']}\")\n",
                "print(f\"  Ground Truth: {sample_preview['ground_truth']}\")\n",
                "print(f\"  Objects ({len(sample_preview['objects'])}): {[o['label'] for o in sample_preview['objects'][:4]]}...\")\n",
                "print(f\"  Relations ({len(sample_preview['relations'])}): {[f'{r[\"subject\"]} {r[\"relation\"]} {r[\"object\"]}' for r in sample_preview['relations'][:3]]}...\")"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 4. Prompt Construction & Output Extraction\n",
                "def format_sg(item):\n",
                "    objects = item.get(\"objects\", [])\n",
                "    relations = item.get(\"relations\", [])\n",
                "    obj_lines = []\n",
                "    for obj in objects:\n",
                "        lbl = obj.get(\"label\", obj.get(\"name\", \"unknown\"))\n",
                "        attrs = obj.get(\"attributes\", [])\n",
                "        attr_str = f\"attributes: [{', '.join(attrs)}]\" if attrs else \"no attributes\"\n",
                "        obj_lines.append(f\"  - {lbl} ({attr_str})\")\n",
                "\n",
                "    rel_lines = []\n",
                "    for rel in relations:\n",
                "        s = rel.get(\"subject\", \"\")\n",
                "        r = rel.get(\"relation\", \"\")\n",
                "        o = rel.get(\"object\", \"\")\n",
                "        rel_lines.append(f'  - \"{s}\" {r} \"{o}\"')\n",
                "        \n",
                "    return len(objects), \"\\n\".join(obj_lines) if obj_lines else \"  None\", len(relations), \"\\n\".join(rel_lines) if rel_lines else \"  None\"\n",
                "\n",
                "def build_direct_prompt(item):\n",
                "    num_obj, obj_str, num_rel, rel_str = format_sg(item)\n",
                "    q = item.get(\"question\", \"\")\n",
                "    return (\n",
                "        \"You are given a scene graph representation of an image containing objects and their relationships, followed by a question.\\n\\n\"\n",
                "        f\"Scene Graph:\\nObjects ({num_obj}):\\n{obj_str}\\n\\n\"\n",
                "        f\"Relationships ({num_rel}):\\n{rel_str}\\n\\n\"\n",
                "        f\"Question: {q}\\n\\n\"\n",
                "        \"Instructions: Based strictly on the scene graph provided above, answer the question accurately. \"\n",
                "        \"Provide only the direct concise answer (a few words or a single term) without explanation or reasoning.\\n\"\n",
                "        \"Answer:\"\n",
                "    )\n",
                "\n",
                "def extract_direct_answer(pred_text):\n",
                "    clean = pred_text.strip()\n",
                "    clean = re.sub(r\"^(?:answer|the answer is|final answer):\\s*\", \"\", clean, flags=re.IGNORECASE).strip()\n",
                "    first_line = clean.split(\"\\n\")[0].strip()\n",
                "    first_line = re.sub(r\"^(?:answer|the answer is|final answer):\\s*\", \"\", first_line, flags=re.IGNORECASE).strip()\n",
                "    return first_line.rstrip(\".\")"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 5. Run Evaluation Loop across 100 Items (~15-20 seconds on T4 GPU)\n",
                "results = []\n",
                "correct_count = 0\n",
                "exact_count = 0\n",
                "cat_stats = {\n",
                "    \"attr_relation\": {\"total\": 0, \"correct\": 0, \"exact\": 0},\n",
                "    \"two_hop\": {\"total\": 0, \"correct\": 0, \"exact\": 0},\n",
                "    \"spatial_conjunction\": {\"total\": 0, \"correct\": 0, \"exact\": 0}\n",
                "}\n",
                "\n",
                "print(\"=\"*75)\n",
                "print(f\"RUNNING DIRECT EVALUATION ON SPATIAL SCENE GRAPHS (ALL {len(SAMPLES)} SAMPLES)\")\n",
                "print(\"=\"*75 + \"\\n\")\n",
                "\n",
                "t_start = time.time()\n",
                "\n",
                "for i, item in enumerate(SAMPLES, 1):\n",
                "    gt = str(item.get(\"ground_truth\", \"\")).strip().lower()\n",
                "    q = item[\"question\"]\n",
                "    t = item[\"template\"]\n",
                "    cat_stats[t][\"total\"] += 1\n",
                "    \n",
                "    prompt = build_direct_prompt(item)\n",
                "    messages = [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": prompt}]}]\n",
                "    \n",
                "    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
                "    inputs = processor(text=[chat_text], return_tensors=\"pt\").to(device)\n",
                "    \n",
                "    start_time = time.time()\n",
                "    with torch.no_grad():\n",
                "        out_ids = model.generate(\n",
                "            **inputs,\n",
                "            max_new_tokens=32,\n",
                "            do_sample=False,\n",
                "            repetition_penalty=1.15\n",
                "        )\n",
                "    elapsed = time.time() - start_time\n",
                "    \n",
                "    gen_ids = out_ids[0][len(inputs[\"input_ids\"][0]):]\n",
                "    raw_pred = processor.decode(gen_ids, skip_special_tokens=True).strip()\n",
                "    parsed = extract_direct_answer(raw_pred)\n",
                "    clean_pred = parsed.lower().rstrip(\".\").strip()\n",
                "    \n",
                "    is_exact = (clean_pred == gt)\n",
                "    is_correct = is_exact or (gt in clean_pred) or (clean_pred in gt and len(clean_pred) > 1)\n",
                "    \n",
                "    if is_correct:\n",
                "        correct_count += 1\n",
                "        cat_stats[t][\"correct\"] += 1\n",
                "    if is_exact:\n",
                "        exact_count += 1\n",
                "        cat_stats[t][\"exact\"] += 1\n",
                "        \n",
                "    status = \"EXACT\" if is_exact else (\"CORRECT\" if is_correct else \"WRONG\")\n",
                "    \n",
                "    results.append({\n",
                "        \"sample_index\": item[\"sample_index\"],\n",
                "        \"image_number\": item[\"image_number\"],\n",
                "        \"template\": t,\n",
                "        \"question\": q,\n",
                "        \"ground_truth\": gt,\n",
                "        \"latency_sec\": elapsed,\n",
                "        \"raw_output\": raw_pred,\n",
                "        \"predicted_answer\": parsed,\n",
                "        \"is_correct\": is_correct,\n",
                "        \"is_exact\": is_exact\n",
                "    })\n",
                "    \n",
                "    if i % 10 == 0 or i == len(SAMPLES) or not is_correct:\n",
                "        print(f\"[{i:03d}/{len(SAMPLES)}] Img {item['image_number']:03d} | {t[:12]:12} | {elapsed:.2f}s | {status:7} | Pred: '{parsed}' | GT: '{gt}'\")\n",
                "\n",
                "total_time = time.time() - t_start\n",
                "acc = (correct_count / len(SAMPLES)) * 100\n",
                "exact_acc = (exact_count / len(SAMPLES)) * 100\n",
                "\n",
                "print(\"\\n\" + \"=\"*75)\n",
                "print(\"BENCHMARK RESULTS (SPATIAL MODEL-GENERATED SCENE GRAPHS)\")\n",
                "print(\"=\"*75)\n",
                "print(f\"Overall Accuracy: {acc:.2f}% ({correct_count}/{len(SAMPLES)})\")\n",
                "print(f\"Exact Match Accuracy: {exact_acc:.2f}% ({exact_count}/{len(SAMPLES)})\")\n",
                "print(f\"Total Inference Time: {total_time:.2f}s (Avg: {total_time/len(SAMPLES):.3f}s/query)\")\n",
                "print(\"\\nCategory Breakdown:\")\n",
                "for cat, st in cat_stats.items():\n",
                "    tot = st[\"total\"]\n",
                "    c_acc = (st[\"correct\"] / tot * 100) if tot > 0 else 0\n",
                "    e_acc = (st[\"exact\"] / tot * 100) if tot > 0 else 0\n",
                "    print(f\"  {cat:20}: {c_acc:6.2f}% Acc ({st['correct']}/{tot}) | Exact: {e_acc:6.2f}% ({st['exact']}/{tot})\")\n",
                "print(\"=\"*75)\n",
                "\n",
                "# Save payload\n",
                "payload = {\n",
                "    \"strategy\": \"Direct Prompting (Spatial Model-Generated Scene Graphs)\",\n",
                "    \"total\": len(SAMPLES),\n",
                "    \"correct\": correct_count,\n",
                "    \"exact\": exact_count,\n",
                "    \"accuracy\": acc,\n",
                "    \"exact_accuracy\": exact_acc,\n",
                "    \"total_time_seconds\": total_time,\n",
                "    \"category_breakdown\": cat_stats,\n",
                "    \"results\": results\n",
                "}\n",
                "with open(\"spatial_sg_benchmark_results.json\", \"w\") as f:\n",
                "    json.dump(payload, f, indent=2)\n",
                "\n",
                "# Copy to Google Drive if mounted\n",
                "for drive_dir in [\"/content/drive/MyDrive/RM\", \"/content/drive/My Drive/RM\"]:\n",
                "    if os.path.exists(drive_dir):\n",
                "        shutil.copy2(\"spatial_sg_benchmark_results.json\", os.path.join(drive_dir, \"spatial_sg_benchmark_results.json\"))\n",
                "        print(f\"[INFO] Copied results to {drive_dir}\")"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 6. Comparison Table with Previous Benchmarks\n",
                "import pandas as pd\n",
                "\n",
                "comparison = {\n",
                "    \"Strategy\": [\n",
                "        \"Ground Truth SGs (Direct)\",\n",
                "        \"Zero-Shot CoT (Images directly)\",\n",
                "        \"Old Model-Generated SGs (Direct)\",\n",
                "        \"NEW Spatial Model-Generated SGs (Direct)\"\n",
                "    ],\n",
                "    \"Overall Acc (%)\": [79.0, 51.0, 25.0, acc],\n",
                "    \"Exact Match (%)\": [77.0, 30.0, 17.0, exact_acc],\n",
                "    \"attr_relation (%)\": [85.3, 67.6, 32.4, (cat_stats['attr_relation']['correct']/cat_stats['attr_relation']['total']*100)],\n",
                "    \"spatial_conjunction (%)\": [84.8, 45.5, 27.3, (cat_stats['spatial_conjunction']['correct']/cat_stats['spatial_conjunction']['total']*100)],\n",
                "    \"two_hop (%)\": [66.7, 39.4, 15.2, (cat_stats['two_hop']['correct']/cat_stats['two_hop']['total']*100)],\n",
                "    \"Avg Latency (s)\": [0.17, 40.2, 0.17, (total_time/len(SAMPLES))]\n",
                "}\n",
                "df = pd.DataFrame(comparison)\n",
                "display(df)"
            ]
        }
    ]
}

local_nb_path = "vlm_evaluations/notebooks/Qwen3_VL_Spatial_SG_Evaluation_Colab.ipynb"
with open(local_nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Created local notebook: {local_nb_path}")

# Sync to Google Drive
drive_nb_path = "G:/My Drive/RM/Qwen3_VL_Spatial_SG_Evaluation_Colab.ipynb"
if os.path.exists("G:/My Drive/RM"):
    shutil.copy2(local_nb_path, drive_nb_path)
    print(f"Synced notebook to Google Drive: {drive_nb_path}")
