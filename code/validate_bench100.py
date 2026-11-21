"""Independent validator for the GQA-100 spatial benchmark (compact layout).

Reads benchmark.json + the per-image files under Images/, Large Scene Graph/,
Small Scene Graph/, Queries/ and re-derives EVERYTHING from scratch.
It does NOT import the builder: derivation logic is re-implemented here, so a
bug in the generator cannot hide behind shared code. The tight subgraph is
recomputed independently from the large graph + anchors.

Gates (per item unless noted):
  (a) every compact edge comes from the tight subgraph (recomputed from the
      large graph: relations touching an anchor)
  (b) every compact edge touches a core node (anchors + query-named objects +
      answer), and the stored core_nodes equal that set
  (c) no duplicate (subject, relation, object) triples in the compact graph
  (d) compact node count (distinct labels) is 4-8, target 5-7; the stored
      nodes list matches the object labels
  (e) the single query's answer re-derived from the compact subgraph, exactly
      one answer
  (f) the query's key edge is present in the compact graph (or the
      between-geometry holds on the compact objects)
  (g) image file exists, is a JPEG (magic FF D8), size > 5 KB
  (h) anchors are labels in the large graph; every compact relation triple
      exists in the large graph; every compact object (label+box) exists in
      the large graph
  (i) between-queries: both anchor labels AND the answer label have exactly
      one instance in the large graph; the centre-x rule is recomputed from
      the compact object boxes and yields exactly the stated answer
  (j) every compact object label is a core node or appears in a compact relation
  (k) benchmark-level: exactly 100 items, unique image_numbers and gqa_image_ids
  (m) mapping consistency: image_number, gqa_image_id and filepath agree across
      the benchmark entry and all three JSON files, filenames match the number,
      the Queries file holds exactly one query object, and it matches the
      benchmark entry's query
  (n) no self-relation edges in the compact graph (subject label == object
      label, e.g. "plate to the left of plate")

Run:  python3 validate_bench100.py
Writes validation_report.json; exits 0 only if every gate passes for every item.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "benchmark.json")
KNOWN_TEMPLATES = {"between", "on", "left_of", "right_of", "with"}


def box_center(b):
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


def query_named_labels(query):
    d = query.get("derivation") or {}
    if query.get("template") in ("left_of", "right_of", "on"):
        return [d.get("object")]
    if query.get("template") == "with":
        return [d.get("subject")]
    return [d.get("anchor_a"), d.get("anchor_b")]


def key_edge(query):
    d = query.get("derivation") or {}
    if query.get("template") in ("left_of", "right_of", "on"):
        return (query.get("answer"), d.get("relation"), d.get("object"))
    if query.get("template") == "with":
        return (d.get("subject"), d.get("relation"), query.get("answer"))
    return None


def rederive(query, comp_objs, comp_rels, full_objs, anchors):
    """Re-derive the answer set for the single query from the compact graph.
    Returns (answers_set, expected_question, error_or_None)."""
    t = query.get("template")
    d = query.get("derivation") or {}
    if t in ("left_of", "right_of", "on"):
        rel, obj = d.get("relation"), d.get("object")
        if not obj or rel not in ("to the left of", "to the right of", "on"):
            return set(), None, f"bad {t} derivation"
        ans = {r["subject"] for r in comp_rels
               if r["relation"] == rel and r["object"] == obj}
        return ans, f"What is {rel} the {obj}?", None
    if t == "with":
        subj = d.get("subject")
        if not subj or d.get("relation") != "with":
            return set(), None, "bad with derivation"
        ans = {r["object"] for r in comp_rels
               if r["relation"] == "with" and r["subject"] == subj}
        return ans, f"What is the {subj} with?", None
    if t == "between":
        a, b = d.get("anchor_a"), d.get("anchor_b")
        if not a or not b:
            return set(), None, "bad between derivation"
        if sorted([a, b]) != sorted(anchors):
            return set(), None, "between anchors != item anchors"
        counts = {}
        for o in full_objs:
            counts[o["label"]] = counts.get(o["label"], 0) + 1
        if counts.get(a, 0) != 1 or counts.get(b, 0) != 1:
            return set(), None, "between anchors not single-instance"
        if counts.get(query.get("answer"), 0) != 1:
            return set(), None, "between answer not single-instance"
        by_label = {o["label"]: o for o in full_objs}
        ax, _ = box_center(by_label[a]["bounding_box"])
        bx, _ = box_center(by_label[b]["bounding_box"])
        lo, hi = min(ax, bx), max(ax, bx)
        if hi <= lo:
            return set(), None, "between anchors share centre-x"
        ans = {o["label"] for o in comp_objs
               if o["label"] not in (a, b)
               and counts.get(o["label"], 0) == 1
               and lo < box_center(o["bounding_box"])[0] < hi}
        return ans, f"What is between the {a} and the {b}?", None
    return set(), None, f"unknown template {t!r}"


def main():
    with open(BENCH) as f:
        bench = json.load(f)
    entries = bench.get("items", [])
    report = {"n_items": len(entries), "gates": {}, "failures": [], "notes": []}

    def fail(label, gate, detail):
        report["failures"].append({"image": label, "gate": gate, "detail": detail})

    gate_k = len(entries) == 100
    if len(entries) != 100:
        fail("BENCHMARK", "k", f"expected 100 items, got {len(entries)}")
    if len({e.get("image_number") for e in entries}) != len(entries):
        gate_k = False
        fail("BENCHMARK", "k", "duplicate image_numbers")
    if len({e.get("gqa_image_id") for e in entries}) != len(entries):
        gate_k = False
        fail("BENCHMARK", "k", "duplicate gqa_image_ids")
    report["gates"]["k_benchmark_100_unique"] = "PASS" if gate_k else "FAIL"

    per_gate_ok = {g: 0 for g in "abcdefghijkmn"}
    per_gate_n = {g: 0 for g in "abcdefghijkmn"}
    per_gate_n["k"] = 1
    per_gate_ok["k"] = 1 if gate_k else 0
    seen_q = set()

    for e in entries:
        num = e.get("image_number")
        iid = e.get("gqa_image_id", "?")
        label = f"GQA Image {num:03d} ({iid})" if isinstance(num, int) else str(iid)
        files = e.get("files") or {}

        # (m) mapping consistency across entry + three JSONs
        per_gate_n["m"] += 1
        m_err = []
        docs = {}
        for kind, key in (("large", "large_scene_graph"), ("small", "small_scene_graph"),
                          ("queries", "queries")):
            path = os.path.join(HERE, files.get(key, ""))
            if not os.path.exists(path):
                m_err.append(f"missing {key} file {files.get(key)}")
                continue
            with open(path) as f:
                docs[kind] = json.load(f)
        for kind, doc in docs.items():
            if doc.get("image_number") != num:
                m_err.append(f"{kind} JSON image_number {doc.get('image_number')} != {num}")
            if doc.get("gqa_image_id") != iid:
                m_err.append(f"{kind} JSON gqa_image_id {doc.get('gqa_image_id')} != {iid}")
            if doc.get("filepath") != e.get("filepath"):
                m_err.append(f"{kind} JSON filepath != entry filepath")
        name = f"GQA Image {num:03d}" if isinstance(num, int) else ""
        for key, suffix in (("image", ".jpg"),
                            ("large_scene_graph", " - Large Scene Graph.json"),
                            ("small_scene_graph", " - Small Scene Graph.json"),
                            ("queries", " - Queries.json")):
            if files.get(key) and not files[key].endswith(name + suffix):
                m_err.append(f"{key} path {files.get(key)} does not match {name + suffix}")
        if "queries" in docs:
            qdoc = docs["queries"]
            if not isinstance(qdoc.get("query"), dict) or isinstance(qdoc.get("query"), list):
                m_err.append("Queries JSON must hold exactly one query object under 'query'")
            elif qdoc["query"].get("question") != (e.get("query") or {}).get("question") or \
                    qdoc["query"].get("answer") != (e.get("query") or {}).get("answer"):
                m_err.append("Queries JSON query != benchmark entry query")
            if qdoc.get("anchors") != e.get("anchors"):
                m_err.append("anchors differ between queries JSON and benchmark entry")
        if "small" in docs and docs["small"].get("anchors") != e.get("anchors"):
            m_err.append("anchors differ between small-graph JSON and benchmark entry")
        if m_err:
            fail(label, "m", "; ".join(m_err))
            continue
        per_gate_ok["m"] += 1

        large, small, qdoc = docs["large"], docs["small"], docs["queries"]
        full_objs, full_rels = large.get("objects", []), large.get("relations", [])
        comp_objs, comp_rels = small.get("objects", []), small.get("relations", [])
        anchors = small.get("anchors") or e.get("anchors") or []
        query = qdoc.get("query") or {}
        comp_labels = {o["label"] for o in comp_objs}
        full_labels = {o["label"] for o in full_objs}
        full_triples = {(r["subject"], r["relation"], r["object"]) for r in full_rels}
        full_obj_keys = {(o["label"], tuple(o["bounding_box"])) for o in full_objs}

        # tight subgraph, recomputed independently from the large graph
        tight_triples = {(r["subject"], r["relation"], r["object"]) for r in full_rels
                         if r["subject"] in anchors or r["object"] in anchors}

        # (d) node count 4-8 (target 5-7); stored nodes list matches labels
        per_gate_n["d"] += 1
        d_err = []
        if not (4 <= len(comp_labels) <= 8):
            d_err.append(f"node count {len(comp_labels)} outside 4-8")
        if set(small.get("nodes") or []) != comp_labels:
            d_err.append("stored nodes list != compact object labels")
        if d_err:
            fail(label, "d", "; ".join(d_err))
        else:
            per_gate_ok["d"] += 1

        # (c) no duplicate triples in the compact graph
        per_gate_n["c"] += 1
        comp_keys = [(r["subject"], r["relation"], r["object"]) for r in comp_rels]
        if len(comp_keys) != len(set(comp_keys)):
            fail(label, "c", f"{len(comp_keys) - len(set(comp_keys))} duplicate triples")
        else:
            per_gate_ok["c"] += 1

        # (n) no self-relation edges in the compact graph
        per_gate_n["n"] += 1
        self_edges = [r for r in comp_rels if r["subject"] == r["object"]]
        if self_edges:
            fail(label, "n", f"{len(self_edges)} self-relation edges, e.g. {self_edges[0]}")
        else:
            per_gate_ok["n"] += 1

        # (a) every compact edge comes from the tight subgraph
        per_gate_n["a"] += 1
        outside = [k for k in comp_keys if k not in tight_triples]
        if outside:
            fail(label, "a", f"{len(outside)} compact edges not in tight subgraph, e.g. {outside[0]}")
        else:
            per_gate_ok["a"] += 1

        # (b) core nodes correct + every compact edge touches a core node
        per_gate_n["b"] += 1
        named = query_named_labels(query)
        expected_core = set(anchors) | set(named) | {query.get("answer")}
        b_err = []
        if set(small.get("core_nodes") or []) != expected_core:
            b_err.append(f"stored core_nodes {sorted(small.get('core_nodes') or [])} "
                         f"!= expected {sorted(expected_core)}")
        untouched = [r for r in comp_rels
                     if r["subject"] not in expected_core and r["object"] not in expected_core]
        if untouched:
            b_err.append(f"{len(untouched)} compact edges touch no core node, e.g. {untouched[0]}")
        if b_err:
            fail(label, "b", "; ".join(b_err))
        else:
            per_gate_ok["b"] += 1

        # (h) anchors in large graph; compact subset of large graph
        per_gate_n["h"] += 1
        h_err = []
        if not anchors or len(anchors) != 2 or any(a not in full_labels for a in anchors):
            h_err.append(f"anchors {anchors} not labels in large graph")
        missing_rels = [k for k in comp_keys if k not in full_triples]
        if missing_rels:
            h_err.append(f"{len(missing_rels)} compact relations not in large graph")
        missing_objs = [o for o in comp_objs
                        if (o["label"], tuple(o["bounding_box"])) not in full_obj_keys]
        if missing_objs:
            h_err.append(f"{len(missing_objs)} compact objects not in large graph")
        if h_err:
            fail(label, "h", "; ".join(h_err))
        else:
            per_gate_ok["h"] += 1

        # (j) every compact object label is a core node or in a compact relation
        per_gate_n["j"] += 1
        rel_labels = {r["subject"] for r in comp_rels} | {r["object"] for r in comp_rels}
        orphan = [l for l in comp_labels if l not in expected_core and l not in rel_labels]
        if orphan:
            fail(label, "j", f"compact object labels with no relation and not core: {orphan}")
        else:
            per_gate_ok["j"] += 1

        # (e) non-empty compact graph, exactly one query
        per_gate_n["e"] += 1
        if len(comp_labels) < 2 or not comp_rels or not query:
            fail(label, "e", f"compact graph {len(comp_labels)} nodes/{len(comp_rels)} rels, "
                             f"query present={bool(query)}")
        else:
            per_gate_ok["e"] += 1

        # (g) image file is a real JPEG
        per_gate_n["g"] += 1
        path = os.path.join(HERE, files.get("image", ""))
        g_err = None
        if not os.path.exists(path):
            g_err = f"missing image file {files.get('image')}"
        else:
            with open(path, "rb") as fh:
                head = fh.read(2)
            size = os.path.getsize(path)
            if head != b"\xff\xd8":
                g_err = f"not a JPEG (magic {head!r})"
            elif size < 5000:
                g_err = f"suspiciously small JPEG ({size} bytes)"
        if g_err:
            fail(label, "g", g_err)
        else:
            per_gate_ok["g"] += 1

        # (e/f/i) the single query: re-derive, key edge, between geometry
        per_gate_n["e"] += 1  # second count: the query itself
        per_gate_n["f"] += 1
        if query.get("template") not in KNOWN_TEMPLATES:
            fail(label, "e", f"unknown template {query.get('template')!r}")
        else:
            ans, expected_q, err = rederive(query, comp_objs, comp_rels, full_objs, anchors)
            if err:
                fail(label, "e", f"Q {query.get('question')!r}: {err}")
            elif query.get("question") != expected_q:
                fail(label, "e", f"question {query.get('question')!r} != template rendering {expected_q!r}")
            elif ans != {query.get("answer")}:
                fail(label, "e", f"Q {query.get('question')!r}: re-derived answers {sorted(ans)} "
                                 f"!= stated answer {query.get('answer')!r}")
            elif query.get("answer") not in comp_labels:
                fail(label, "e", f"answer {query.get('answer')!r} not a compact object label")
            else:
                per_gate_ok["e"] += 1
                if query.get("template") == "between":
                    per_gate_n["i"] += 1
                    per_gate_ok["i"] += 1
                ke = key_edge(query)
                if ke is None:
                    per_gate_ok["f"] += 1  # between: geometry already re-derived above
                elif any((r["subject"], r["relation"], r["object"]) == ke for r in comp_rels):
                    per_gate_ok["f"] += 1
                else:
                    fail(label, "f", f"key edge {ke} not in compact graph")

        # duplicate (image_number, question) pairs
        key = (num, query.get("question"))
        if key in seen_q:
            fail(label, "k", f"duplicate question {query.get('question')!r}")
        seen_q.add(key)

    for g in "abcdefghijkmn":
        n, ok = per_gate_n[g], per_gate_ok[g]
        report["gates"][f"{g}_pass"] = f"{ok}/{n}"
    report["all_pass"] = (not report["failures"] and gate_k
                          and all(per_gate_ok[g] == per_gate_n[g] for g in "abcdefghijkmn"
                                  if per_gate_n[g] > 0))
    with open(os.path.join(HERE, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=1)

    print(f"items: {len(entries)}")
    for g in "abcdefghijkmn":
        print(f"  gate {g}: {per_gate_ok[g]}/{per_gate_n[g]}")
    print(f"  gate k: {'PASS' if gate_k else 'FAIL'}")
    print(f"  total failures: {len(report['failures'])}")
    for fl in report["failures"][:20]:
        print(f"    FAIL {fl['image']} gate {fl['gate']}: {fl['detail']}")
    print("ALL PASS" if report["all_pass"] else "VALIDATION FAILED")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
