"""Prototype: GQA scene graph -> local subgraph -> spatial query + ground-truth answer.
Run: python3 gqa_bench_prototype.py
Input: gqa-example-2395226-graph.json (real GQA graph, 23 objects, 67 relations).
Steps: pick 2 anchor objects -> local subgraph (anchors + direct neighbours +
relations among them) -> generate queries from templates -> keep only queries
with exactly ONE answer inside the subgraph.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
graph = json.load(open(os.path.join(HERE, "gqa-example-2395226-graph.json")))
objects = graph["objects"]          # [{label, attributes, bounding_box}]
relations = graph["relations"]      # [{subject, relation, object}]  (label-based)

def box_center(b):
    x, y, w, h = b
    return (x + w / 2, y + h / 2)

def local_subgraph(anchor_a, anchor_b):
    """Anchors + every object directly related to either anchor, and all
    relations whose subject and object are both inside that set."""
    keep = {anchor_a, anchor_b}
    for r in relations:
        if r["subject"] in keep or r["object"] in keep:
            keep.add(r["subject"]); keep.add(r["object"])
    sub_rels = [r for r in relations if r["subject"] in keep and r["object"] in keep]
    sub_objs = [o for o in objects if o["label"] in keep]
    return {"anchors": [anchor_a, anchor_b], "objects": sub_objs, "relations": sub_rels}

def unique_answers(pred):
    """pred(rel) -> list of (subject, relation, object) matching; return answers
    only when exactly one distinct answer label exists."""
    hits = [r for r in relations if pred(r)]
    answers = sorted({r["object"] if pred.__name__ == "by_object" else r["subject"] for r in hits})
    return answers

def by_object(r, rel, obj):
    return r["relation"] == rel and r["object"] == obj
def by_subject(r, rel, subj):
    return r["relation"] == rel and r["subject"] == subj

queries = []
def add_q(template_q, answers, how):
    if len(answers) == 1:
        queries.append({"question": template_q, "answer": answers[0], "from": how})

# --- anchor pair: tomato + fork (both well-connected in this graph) ---
sub = local_subgraph("tomato", "fork")
print(f"SUBGRAPH: {len(sub['objects'])} objects, {len(sub['relations'])} relations")
print("objects:", sorted({o['label'] for o in sub['objects']}))

# Q1: what is to the left of X?  (subjects S with S --left of--> X)
for obj in ["tomato", "fork", "plate"]:
    hits = sorted({r["subject"] for r in sub["relations"] if r["relation"] == "to the left of" and r["object"] == obj})
    add_q(f"What is to the left of the {obj}?", hits, "relation to the left of")
# Q2: what is on X?
for obj in ["plate", "table"]:
    hits = sorted({r["subject"] for r in sub["relations"] if r["relation"] == "on" and r["object"] == obj})
    add_q(f"What is on the {obj}?", hits, "relation on")
# Q3: what is X with?
for obj in ["sandwich"]:
    hits = sorted({r["object"] for r in sub["relations"] if r["relation"] == "with" and r["subject"] == obj})
    add_q(f"What is the {obj} with?", hits, "relation with")
# Q4: what is BETWEEN A and B? (derived from boxes, not a GQA relation)
by_label = {}
for o in objects:
    by_label.setdefault(o["label"], o)
for a, b in [("tomato", "fork"), ("carrots", "sandwich")]:
    if a not in by_label or b not in by_label:
        continue
    ax, _ = box_center(by_label[a]["bounding_box"])
    bx, _ = box_center(by_label[b]["bounding_box"])
    lo, hi = min(ax, bx), max(ax, bx)
    between = sorted({o["label"] for o in objects
                      if o["label"] not in (a, b) and lo < box_center(o["bounding_box"])[0] < hi})
    add_q(f"What is between the {a} and the {b}?", between, "bounding-box geometry")

print(f"\nGENERATED {len(queries)} UNIQUE-ANSWER QUERIES:")
for q in queries:
    print(f"  Q: {q['question']}  ->  A: {q['answer']}   [{q['from']}]")

out = {"image": graph["image"], "subgraph": sub, "queries": queries}
json.dump(out, open(os.path.join(HERE, "gqa-bench-prototype.json"), "w"), indent=1)
print("\nSaved: gqa-bench-prototype.json")
