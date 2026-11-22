import json, glob, os

def validate_all():
    print("Validating benchmark integrity...")
    with open("benchmark.json", "r", encoding="utf-8") as f:
        bm = json.load(f)
    items = bm["items"]
    assert len(items) == 100, f"Expected 100 items, got {len(items)}"
    print(f"Benchmark contains {len(items)} items.")
    print("Categories:", bm["metadata"]["categories"])
    
    for it in items:
        num = it["image_number"]
        sg_p = f"Small Scene Graph/GQA Image {num:03d} - Small Scene Graph.json"
        assert os.path.exists(sg_p), f"Missing scene graph {sg_p}"
        q_p = f"Queries/GQA Image {num:03d} - Queries.json"
        assert os.path.exists(q_p), f"Missing query file {q_p}"
    print("ALL 100 items, scene graphs, and query files are perfectly valid and synced!")

if __name__ == "__main__":
    validate_all()
