"""One-image walkthrough: full GQA graph -> tight local subgraph -> generated queries.
Anchors: tomato + fork. Tight = only relations touching an anchor (the fix flagged
in the prototype, where the subgraph ballooned to 20/23 objects).
Writes gqa-walkthrough.json and gqa-walkthrough.html next to this script."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
graph = json.load(open(os.path.join(HERE, "gqa-example-2395226-graph.json")))
objects, relations = graph["objects"], graph["relations"]
ANCHORS = ["tomato", "fork"]

def box_center(b):
    x, y, w, h = b
    return (x + w / 2, y + h / 2)

# ---- Stage 2: tight subgraph (anchor-only relations) ----
sub_rels = [r for r in relations if r["subject"] in ANCHORS or r["object"] in ANCHORS]
keep_labels = set(ANCHORS)
for r in sub_rels:
    keep_labels.add(r["subject"]); keep_labels.add(r["object"])
sub_objs = [o for o in objects if o["label"] in keep_labels]

# ---- Stage 3: queries from the TIGHT subgraph, unique-answer only ----
queries = []
def add_q(q, answers, how):
    if len(answers) == 1:
        queries.append({"question": q, "answer": answers[0], "from": how})
for obj in ["tomato", "fork", "plate"]:
    hits = sorted({r["subject"] for r in sub_rels if r["relation"] == "to the left of" and r["object"] == obj})
    add_q(f"What is to the left of the {obj}?", hits, "relation: to the left of")
for obj in ["plate", "table"]:
    hits = sorted({r["subject"] for r in sub_rels if r["relation"] == "on" and r["object"] == obj})
    add_q(f"What is on the {obj}?", hits, "relation: on")
for obj in ["sandwich"]:
    hits = sorted({r["object"] for r in sub_rels if r["relation"] == "with" and r["subject"] == obj})
    add_q(f"What is the {obj} with?", hits, "relation: with")
by_label = {}
for o in objects:
    by_label.setdefault(o["label"], o)
for a, b in [("tomato", "fork")]:
    ax, _ = box_center(by_label[a]["bounding_box"]); bx, _ = box_center(by_label[b]["bounding_box"])
    lo, hi = min(ax, bx), max(ax, bx)
    between = sorted({o["label"] for o in sub_objs if o["label"] not in (a, b) and lo < box_center(o["bounding_box"])[0] < hi})
    add_q(f"What is between the {a} and the {b}?", between, "bounding-box geometry (not a GQA relation)")

walk = {
  "image": graph["image"], "source": graph["source"], "anchors": ANCHORS,
  "full": {"n_objects": len(objects), "n_relations": len(relations), "objects": objects, "relations": relations},
  "subgraph": {"n_objects": len(sub_objs), "n_relations": len(sub_rels), "objects": sub_objs, "relations": sub_rels},
  "queries": queries,
}
json.dump(walk, open(os.path.join(HERE, "gqa-walkthrough.json"), "w"), indent=1)
print(f"FULL: {len(objects)} objects, {len(relations)} relations")
print(f"SUBGRAPH (anchors {ANCHORS}): {len(sub_objs)} objects, {len(sub_rels)} relations")
print("subgraph objects:", sorted({o['label'] for o in sub_objs}))
print(f"QUERIES: {len(queries)}")
for q in queries:
    print(f"  Q: {q['question']} -> A: {q['answer']} [{q['from']}]")

# ---- HTML walkthrough (stage 1/2/3) with an SVG of the subgraph ----
W, H = 500, 375
def sx(x): return round(x * W)
def sy(y): return round(y * H)
nodes = {}
for o in sub_objs:
    cx, cy = box_center(o["bounding_box"])
    nodes.setdefault(o["label"], (sx(cx), sy(cy)))
edges_svg = ""
for r in sub_rels:
    x1, y1 = nodes[r["subject"]]; x2, y2 = nodes[r["object"]]
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    edges_svg += (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#94a3b8" stroke-width="1"/>'
                  f'<text x="{mx}" y="{my}" font-size="8" fill="#475569">{r["relation"]}</text>')
nodes_svg = ""
for label, (x, y) in nodes.items():
    fill = "#dc2626" if label in ANCHORS else "#2563eb"
    nodes_svg += (f'<circle cx="{x}" cy="{y}" r="13" fill="{fill}"/>'
                  f'<text x="{x}" y="{y+3}" font-size="9" fill="#fff" text-anchor="middle">{label[:8]}</text>')
obj_rows = "".join(
    f"<tr><td>{o['label']}</td><td>{', '.join(o.get('attributes', [])) or '-'}</td>"
    f"<td>[{', '.join(f'{v:.2f}' for v in o['bounding_box'])}]</td></tr>" for o in objects)
rel_rows = "".join(f"<tr><td>{r['subject']}</td><td>{r['relation']}</td><td>{r['object']}</td></tr>" for r in relations)
sub_rel_rows = "".join(f"<tr><td>{r['subject']}</td><td>{r['relation']}</td><td>{r['object']}</td></tr>" for r in sub_rels)
q_rows = "".join(f"<tr><td>{q['question']}</td><td><b>{q['answer']}</b></td><td>{q['from']}</td></tr>" for q in queries)
html = f"""<div style="font-family:system-ui,sans-serif;max-width:640px;margin:0 auto;color:#0f172a">
<h2 style="font-size:18px">GQA walkthrough — 1 image, 3 stages</h2>
<p style="font-size:13px;color:#475569">Image: <b>{graph['image']}</b> (food table) · Source: Voxel51/GQA-Scene-Graph · Anchors: <b>tomato + fork</b></p>
<h3 style="font-size:15px">Stage 1 — Full original graph: {len(objects)} objects, {len(relations)} relations</h3>
<details><summary style="cursor:pointer;font-size:13px">Objects ({len(objects)}) — click to open</summary>
<table style="font-size:12px;border-collapse:collapse"><tr><th style="text-align:left">label</th><th style="text-align:left">attributes</th><th style="text-align:left">box [x,y,w,h]</th></tr>{obj_rows}</table></details>
<details><summary style="cursor:pointer;font-size:13px">Relations ({len(relations)}) — click to open</summary>
<table style="font-size:12px;border-collapse:collapse"><tr><th style="text-align:left">subject</th><th style="text-align:left">relation</th><th style="text-align:left">object</th></tr>{rel_rows}</table></details>
<h3 style="font-size:15px">Stage 2 — Tight subgraph: {len(sub_objs)} objects, {len(sub_rels)} relations</h3>
<p style="font-size:12px;color:#475569">Rule: only relations touching an anchor (tomato or fork). Red = anchor, blue = neighbour. Node position = bounding-box centre.</p>
<svg width="{W}" height="{H}" style="border:1px solid #cbd5e1;background:#f8fafc">{edges_svg}{nodes_svg}</svg>
<details><summary style="cursor:pointer;font-size:13px">Subgraph relations ({len(sub_rels)}) — click to open</summary>
<table style="font-size:12px;border-collapse:collapse"><tr><th style="text-align:left">subject</th><th style="text-align:left">relation</th><th style="text-align:left">object</th></tr>{sub_rel_rows}</table></details>
<h3 style="font-size:15px">Stage 3 — Questions from the subgraph ({len(queries)})</h3>
<table style="font-size:13px;border-collapse:collapse"><tr><th style="text-align:left">question</th><th style="text-align:left">answer</th><th style="text-align:left">derived from</th></tr>{q_rows}</table>
<p style="font-size:12px;color:#475569">Only questions with exactly one answer inside the subgraph are kept; the rest are dropped as ambiguous.</p>
</div>"""
open(os.path.join(HERE, "gqa-walkthrough.html"), "w").write(html)
print("WROTE gqa-walkthrough.json + gqa-walkthrough.html")
