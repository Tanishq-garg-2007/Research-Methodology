import json, re, shutil, os
from collections import Counter

JSON_PATH = "kaggle_spatial_sgg_100/output/scene_graphs_100_images_spatial.json"
DRIVE_PATH = "G:/My Drive/RM/scene_graphs_100_images_spatial.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

def repair_truncated_json(raw_text):
    # strip markdown markers if present
    m = re.search(r"```json\s*(.*)", raw_text, re.DOTALL)
    target = m.group(1) if m else raw_text
    target = target.rstrip().rstrip("`")
    
    # backtrack to last complete item '}'
    last_brace = target.rfind("}")
    if last_brace == -1:
        return None
        
    candidate = target[:last_brace+1]
    
    # Balance brackets
    open_brackets = candidate.count("[") - candidate.count("]")
    open_braces = candidate.count("{") - candidate.count("}")
    
    candidate += ("]" * max(0, open_brackets))
    candidate += ("}" * max(0, open_braces))
    
    # Remove trailing commas
    candidate = re.sub(r",\s*\]", "]", candidate)
    candidate = re.sub(r",\s*\}", "}", candidate)
    
    try:
        return json.loads(candidate)
    except Exception:
        # Try finding the substring between the first { and the balanced closing
        first_brace = candidate.find("{")
        if first_brace != -1:
            try:
                return json.loads(candidate[first_brace:])
            except Exception:
                return None
    return None

repaired = 0
for item in data:
    if not item.get("is_valid_json"):
        raw = item["scene_graph"].get("raw", "")
        repaired_sg = repair_truncated_json(raw)
        if repaired_sg and isinstance(repaired_sg, dict):
            item["scene_graph"] = repaired_sg
            item["is_valid_json"] = True
            item["repaired"] = True
            item["objects_count"] = len(repaired_sg.get("objects", []))
            item["relations_count"] = len(repaired_sg.get("relations", []))
            
            directional_preds = {"to the left of", "to the right of", "above", "below", "in front of", "behind", "next to", "beside", "near", "on top of"}
            item["directional_relations_count"] = sum(1 for r in repaired_sg.get("relations", []) if str(r.get("relation", "")).lower().strip() in directional_preds)
            repaired += 1

print(f"Successfully repaired {repaired} / 24 truncated scene graphs!")
valid_now = sum(1 for d in data if d.get("is_valid_json"))
print(f"Total valid scene graphs now: {valid_now} / {len(data)} ({valid_now/len(data)*100:.1f}%)")

# Recount relations and predicates
all_rels = []
pred_counter = Counter()
for item in data:
    if item.get("is_valid_json"):
        for r in item["scene_graph"].get("relations", []):
            pred = str(r.get("relation", "")).lower().strip()
            pred_counter[pred] += 1
            all_rels.append(r)

directional_preds = {
    "to the left of", "to the right of", "above", "below",
    "in front of", "behind", "next to", "beside", "near",
    "on top of", "under"
}

total_dir = sum(count for pred, count in pred_counter.items() 
                if pred in directional_preds or any(d in pred for d in ["left", "right", "above", "below", "front", "behind", "beside"]))

print(f"Total relations extracted across 100 images: {len(all_rels)}")
print(f"Total directional spatial relations: {total_dir} ({total_dir/len(all_rels)*100:.1f}%)")

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

if os.path.exists("G:/My Drive/RM"):
    shutil.copy2(JSON_PATH, DRIVE_PATH)
    print(f"[INFO] Synced finalized scene graphs to Drive: {DRIVE_PATH}")
