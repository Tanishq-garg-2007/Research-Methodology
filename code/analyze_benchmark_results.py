import json, os
import pandas as pd

def analyze(direct_path="vlm_evaluations/prompting_experiments/results_zero_shot_direct.json",
            cot_path="vlm_evaluations/prompting_experiments/results_zero_shot_cot.json"):
    if not os.path.exists(direct_path) or not os.path.exists(cot_path):
        print("Results files not found.")
        return
    with open(direct_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    with open(cot_path, "r", encoding="utf-8") as f:
        c = json.load(f)
        
    print("="*75)
    print("BENCHMARK COMPARISON SCOREBOARD (100 GQA SAMPLES):")
    print("="*75)
    rows = []
    for name, data in [("Direct (Primary)", d), ("Zero-Shot CoT", c)]:
        bd = data.get("category_breakdown", {})
        
        def calc_acc(cat):
            st = bd.get(cat, {})
            tot = st.get("total", 0)
            corr = st.get("correct", 0)
            return f"{corr/tot*100:.1f}% ({corr:02d}/{tot:02d})" if tot else "N/A"

        rows.append({
            "Strategy": name,
            "Overall Acc": f"{data.get('accuracy', 0)*100:.1f}% ({data.get('correct', 0)}/{data.get('total', 0)})",
            "Exact Match": f"{data.get('exact_accuracy', 0)*100:.1f}% ({data.get('exact', 0)}/{data.get('total', 0)})",
            "attr_relation": calc_acc("attr_relation"),
            "two_hop": calc_acc("two_hop"),
            "spatial_conjunction": calc_acc("spatial_conjunction"),
            "Runtime": f"{data.get('total_time_seconds', 0):.1f}s"
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("="*75)

if __name__ == "__main__":
    analyze()
