import json, shutil, os
import pandas as pd

with open('kaggle_gen_sg_results/generated_sg_benchmark_results.json', 'r', encoding='utf-8') as f:
    gen_data = json.load(f)

with open('vlm_evaluations/prompting_experiments/results_zero_shot_direct.json', 'r', encoding='utf-8') as f:
    gt_direct = json.load(f)

with open('vlm_evaluations/prompting_experiments/results_zero_shot_cot.json', 'r', encoding='utf-8') as f:
    gt_cot = json.load(f)

print("="*80)
print("RESULTS: DIRECT PROMPTING ON MODEL-GENERATED SCENE GRAPHS")
print("="*80)
total = gen_data['total']
corr = gen_data['correct']
exact = gen_data['exact']
acc = gen_data['accuracy'] * 100
ex_acc = gen_data['exact_accuracy'] * 100
tot_time = gen_data['total_time_seconds']
avg_time = gen_data['avg_time_per_sample']

print(f"Overall Accuracy: {corr}/{total} ({acc:.2f}%)")
print(f"Exact Matches:    {exact}/{total} ({ex_acc:.2f}%)")
print(f"Total Runtime:    {tot_time:.2f}s ({avg_time:.3f}s/sample)")

print("\n" + "-"*60)
print("CATEGORY BREAKDOWN (MODEL-GENERATED SCENE GRAPHS):")
print("-"*60)
for cat, st in gen_data['category_breakdown'].items():
    t = st['total']
    c = st['correct']
    e = st['exact']
    c_acc = c / t * 100 if t else 0
    e_acc = e / t * 100 if t else 0
    print(f"  - {cat:<22}: {c:2d}/{t:2d} ({c_acc:5.2f}% Acc | {e_acc:5.2f}% Exact)")
print("-"*60)

print("\n" + "="*80)
print("HEAD-TO-HEAD COMPARISON: GROUND TRUTH SCENE GRAPH vs MODEL-GENERATED SCENE GRAPH")
print("="*80)
rows = []
for name, data in [("GT Scene Graph (Direct)", gt_direct),
                   ("Model-Generated SG (Direct)", gen_data),
                   ("GT Scene Graph (CoT)", gt_cot)]:
    bd = data.get("category_breakdown", {})
    def get_c(cat):
        st = bd.get(cat, {})
        tot = st.get('total', 0)
        c = st.get('correct', 0)
        return f"{c/tot*100:5.1f}% ({c:2d}/{tot:2d})" if tot else "N/A"

    rows.append({
        "Scene Graph Source & Strategy": name,
        "Overall Acc": f"{data.get('accuracy', 0)*100:5.1f}% ({data.get('correct', 0):2d}/{data.get('total', 0)})",
        "Exact Match": f"{data.get('exact_accuracy', 0)*100:5.1f}% ({data.get('exact', 0):2d}/{data.get('total', 0)})",
        "attr_relation": get_c("attr_relation"),
        "two_hop": get_c("two_hop"),
        "spatial_conjunction": get_c("spatial_conjunction"),
        "Time (s)": f"{data.get('total_time_seconds', 0):6.1f}s"
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print("="*80)

# Inspect errors and sample correct items
results = gen_data['results']
correct_items = [r for r in results if r['is_correct']]
incorrect_items = [r for r in results if not r['is_correct']]

print(f"\nCorrect Samples Count: {len(correct_items)}")
print(f"Incorrect Samples Count: {len(incorrect_items)}")

print("\nSample Successful Detections (First 5):")
for r in correct_items[:5]:
    print(f"  Img {r['image_number']:03d} [{r['template']}] | Q: {r['question']} | GT: '{r['ground_truth']}' | Pred: '{r['predicted_answer']}'")

print("\nSample Error Cases (First 5):")
for r in incorrect_items[:5]:
    print(f"  Img {r['image_number']:03d} [{r['template']}] | Q: {r['question']} | GT: '{r['ground_truth']}' | Pred: '{r['predicted_answer']}'")

# Copy to Drive
drive_dir = 'G:/My Drive/RM'
if os.path.exists(drive_dir):
    shutil.copy('kaggle_gen_sg_results/generated_sg_benchmark_results.json', os.path.join(drive_dir, 'generated_sg_benchmark_results.json'))
    print("\nSynced generated_sg_benchmark_results.json to Google Drive!")
