import json, os, shutil
from collections import Counter

JSON_PATH = "kaggle_spatial_sgg_100/output/scene_graphs_100_images_spatial.json"
DRIVE_PATH = "G:/My Drive/RM/scene_graphs_100_images_spatial.json"

if not os.path.exists(JSON_PATH):
    print(f"Error: {JSON_PATH} not found!")
    exit(1)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} items from {JSON_PATH}")

total_images = len(data)
valid_json_count = sum(1 for item in data if item.get("is_valid_json", False))
total_latency = sum(item.get("latency_sec", 0) for item in data)
avg_latency = total_latency / max(total_images, 1)

all_objects = []
all_relations = []
predicate_counter = Counter()

directional_preds = {
    "to the left of", "to the right of", "above", "below",
    "in front of", "behind", "next to", "beside", "near",
    "on top of", "under"
}

images_with_spatial_rels = 0

for item in data:
    sg = item.get("scene_graph", {})
    objs = sg.get("objects", [])
    rels = sg.get("relations", [])
    
    all_objects.extend(objs)
    all_relations.extend(rels)
    
    has_dir = False
    for r in rels:
        pred = str(r.get("relation", "")).lower().strip()
        predicate_counter[pred] += 1
        if pred in directional_preds or any(d in pred for d in ["left", "right", "above", "below", "front", "behind", "beside"]):
            has_dir = True
            
    if has_dir:
        images_with_spatial_rels += 1

total_directional = sum(count for pred, count in predicate_counter.items() 
                        if pred in directional_preds or any(d in pred for d in ["left", "right", "above", "below", "front", "behind", "beside"]))

directional_pct = (total_directional / max(len(all_relations), 1)) * 100

print("\n" + "="*70)
print("SPATIAL SCENE GRAPH GENERATION (100 IMAGES) ANALYSIS")
print("="*70)
print(f"Total Images: {total_images} / 100")
print(f"Valid JSON Syntax: {valid_json_count} / {total_images} ({valid_json_count/total_images*100:.1f}%)")
print(f"Total Generation Time: {total_latency:.1f}s ({total_latency/60:.1f} min)")
print(f"Average Time per Image: {avg_latency:.2f}s")
print(f"Total Objects Extracted: {len(all_objects)} (Avg: {len(all_objects)/max(total_images,1):.1f} per image)")
print(f"Total Relations Extracted: {len(all_relations)} (Avg: {len(all_relations)/max(total_images,1):.1f} per image)")
print(f"Images with Directional Spatial Relations: {images_with_spatial_rels} / {total_images} ({images_with_spatial_rels/total_images*100:.1f}%)")
print(f"Total Directional Spatial Triples: {total_directional} / {len(all_relations)} ({directional_pct:.1f}%)")
print("\nTop 15 Most Frequent Predicates in New SGs:")
for pred, count in predicate_counter.most_common(15):
    is_spatial = "✓ [SPATIAL]" if (pred in directional_preds or any(d in pred for d in ["left", "right", "above", "below", "front", "behind", "beside"])) else " "
    print(f"  {is_spatial:12} '{pred}': {count} ({count/len(all_relations)*100:.1f}%)")
print("="*70)

# Sync to Google Drive
if os.path.exists("G:/My Drive/RM"):
    shutil.copy2(JSON_PATH, DRIVE_PATH)
    print(f"\n[INFO] Successfully synced to Google Drive: {DRIVE_PATH}")
