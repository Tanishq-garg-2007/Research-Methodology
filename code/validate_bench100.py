"""Independent validator for the GQA-100 spatial benchmark (compact layout).

Reads benchmark.json + the per-image files under Images/, Large Scene Graph/,
Small Scene Graph/, Queries/ and re-derives EVERYTHING from scratch.
It does NOT import the builder or the v2 generator: derivation logic is
re-implemented here, so a bug in the generator cannot hide behind shared
code. The tight subgraph is recomputed independently from the large graph +
anchors.

v2 (2026-11-21): the queries were rebuilt with harder templates
(two_hop, between_attr, counting, attr_relation) from the FROZEN small
graphs. Gate changes vs v1 (justified in VALIDATION.md):
  (b) stored core_nodes still describes the v1 build's core (the small-graph
      files are frozen, so it cannot equal the v2 query's core). The gate now
      checks: every compact edge touches a STORED core node; stored
      core_nodes are a subset of the compact labels; and the v2 query's
      named labels (+ answer label, except counting's integer answer) are
      all compact labels -- i.e. the query is answerable from the stored
      graph.
  (e) extended to the four v2 templates. Counting answers are integers
      re-derived from the LARGE graph (not the compact graph); the "answer
      is a compact label" check is skipped for counting.
  (f) two_hop requires BOTH hop edges present in the compact graph;
      between_attr/counting have no key edge (geometry/count, re-derived
      in gate e), as between did in v1.
  (i) extended to between_attr: the attribute-qualified anchor must select
      exactly one instance in the large graph, the other anchor must be
      single-instance or likewise attribute-pinned, the answer must be
      single-instance, and the centre-x rule is recomputed from the compact
      object boxes and yields exactly the stated answer (the v1 between
      rule).

Gates (per item unless noted):
  (a) every compact edge comes from the tight subgraph (recomputed from the
      large graph: relations touching an anchor)
  (b) see v2 note above
  (c) no duplicate (subject, relation, object) triples in the compact graph
  (d) compact node count (distinct labels) is 4-8, target 5-7; the stored
      nodes list matches the object labels
  (e) the single query's answer re-derived from the compact graph (counting:
      from the large graph), exactly one answer
  (f) the query's key edge(s) present in the compact graph (or the
      between-geometry holds / the count is re-derived)
  (g) image file exists, is a JPEG (magic FF D8), size > 5 KB
  (h) anchors are labels in the large graph; every compact relation triple
      exists in the large graph; every compact object (label+box) exists in
      the large graph
  (i) between-queries: anchors + answer single-instance (or attribute-
      pinned); the centre-x rule recomputed and yields exactly the answer
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
KNOWN_TEMPLATES = {"between", "on", "left_of", "right_of", "with",
                   "two_hop", "between_attr", "counting", "attr_relation"}

IRREGULAR_PLURALS = {
    "man": "men", "woman": "women", "child": "children",
    "person": "people", "foot": "feet", "tooth": "teeth",
    "mouse": "mice", "goose": "geese",
}


def box_center(b):
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


def pluralize(noun):
    """Shared rendering rule (duplicated from generate_queries_v2.py, which
    this validator does not import). GQA labels ending in 's' are usually
    ALREADY plural (chairs, decorations, glasses, clothes), so they are
    returned unchanged -- except 'ss'/'us' endings, which are singular
    (glass -> glasses, bus -> buses)."""
    if noun in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[noun]
    if noun.endswith("ss") or noun.endswith("us"):
        return noun + "es"
    if noun.endswith("s"):
        return noun
    if noun.endswith(("x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("o"):
        return noun + "es"
    if len(noun) > 1 and noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def attr_phrase(label, attrs):
    if attrs:
        return ", ".join(attrs) + " " + label
    return label


def find_pinned_instance(label, attrs, full_objs):
    """The unique large-graph instance of `label` carrying all of `attrs`
    (attrs empty => the label must be single-instance). None otherwise."""
    cands = [o for o in full_objs if o["label"] == label
             and set(attrs) <= set(o.get("attributes") or [])]
    if len(cands) == 1:
        return cands[0]
    return None


def query_named_labels(query):
    d = query.get("derivation") or {}
    t = query.get("template")
    if t == "two_hop":
        return [d.get("intermediate"), d.get("hop1_subject")]
    if t == "between_attr":
        return [d.get("anchor_a"), d.get("anchor_b")]
    if t == "counting":
        return [d.get("counted_label"), d.get("anchor")]
    if t == "attr_relation":
        if d.get("relation") == "with":
            return [d.get("subject")]
        return [d.get("object")]
    if t in ("left_of", "right_of", "on"):
        return [d.get("object")]
    if t == "with":
        return [d.get("subject")]
    return [d.get("anchor_a"), d.get("anchor_b")]  # between


def key_edges(query):
    """(subject, relation, object) triples that must be present in the
    compact graph. Empty for geometry/count templates."""
    d = query.get("derivation") or {}
    t = query.get("template")
    if t == "two_hop":
        return [(d.get("hop1_subject"), d.get("hop1_relation"), d.get("intermediate")),
                (query.get("answer"), d.get("hop2_relation"), d.get("intermediate"))]
    if t == "attr_relation":
        if d.get("relation") == "with":
            return [(d.get("subject"), "with", query.get("answer"))]
        return [(query.get("answer"), d.get("relation"), d.get("object"))]
    if t in ("left_of", "right_of", "on"):
        return [(query.get("answer"), d.get("relation"), d.get("object"))]
    if t == "with":
        return [(d.get("subject"), d.get("relation"), query.get("answer"))]
    return []


def rederive(query, comp_objs, comp_rels, full_objs, anchors):
    """Re-derive the answer for the single query.
    Returns (answers, expected_question, error_or_None). answers is a set
    of labels, except for counting where it is an int."""
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
    if t == "two_hop":
        s1, rel1 = d.get("hop1_subject"), d.get("hop1_relation")
        b, rel2 = d.get("intermediate"), d.get("hop2_relation")
        if not s1 or not b or rel1 not in ("on", "to the left of",
                                           "to the right of", "with") or \
                rel2 not in ("to the left of", "to the right of", "on"):
            return set(), None, "bad two_hop derivation"
        hop1 = (s1, rel1, b)
        if not any((r["subject"], r["relation"], r["object"]) == hop1
                   for r in comp_rels):
            return set(), None, f"hop-1 edge {hop1} not in compact graph"
        ans = {r["subject"] for r in comp_rels
               if r["relation"] == rel2 and r["object"] == b}
        if rel1 == "on":
            mid = f"the {b} that the {s1} is on"
        elif rel1 == "to the left of":
            mid = f"the {b} that is to the right of the {s1}"
        elif rel1 == "to the right of":
            mid = f"the {b} that is to the left of the {s1}"
        else:
            mid = f"the {b} that is with the {s1}"
        return ans, f"What is {rel2} {mid}?", None
    if t == "attr_relation":
        rel = d.get("relation")
        if rel not in ("to the left of", "to the right of", "on", "with"):
            return set(), None, "bad attr_relation derivation"
        if rel == "with":
            anchor = d.get("subject")
            if not anchor:
                return set(), None, "bad attr_relation derivation"
            ans = {r["object"] for r in comp_rels
                   if r["relation"] == "with" and r["subject"] == anchor}
            q = f"What is the {attr_phrase(anchor, d.get('anchor_attributes') or [])} with?"
        else:
            anchor = d.get("object")
            if not anchor:
                return set(), None, "bad attr_relation derivation"
            ans = {r["subject"] for r in comp_rels
                   if r["relation"] == rel and r["object"] == anchor}
            q = f"What is {rel} the {attr_phrase(anchor, d.get('anchor_attributes') or [])}?"
        return ans, q, None
    if t == "between_attr":
        a, b = d.get("anchor_a"), d.get("anchor_b")
        if not a or not b:
            return set(), None, "bad between_attr derivation"
        a_inst = find_pinned_instance(a, d.get("anchor_a_attributes") or [], full_objs)
        b_inst = find_pinned_instance(b, d.get("anchor_b_attributes") or [], full_objs)
        if a_inst is None or b_inst is None:
            return set(), None, "between_attr anchor not uniquely pinnable"
        ax, _ = box_center(a_inst["bounding_box"])
        bx, _ = box_center(b_inst["bounding_box"])
        lo, hi = min(ax, bx), max(ax, bx)
        if hi <= lo:
            return set(), None, "between_attr anchors share centre-x"
        ans = {o["label"] for o in comp_objs
               if o["label"] not in (a, b)
               and lo < box_center(o["bounding_box"])[0] < hi}
        q = (f"What is between the "
             f"{attr_phrase(a, d.get('anchor_a_attributes') or [])} and the "
             f"{attr_phrase(b, d.get('anchor_b_attributes') or [])}?")
        return ans, q, None
    if t == "counting":
        lab, rel = d.get("counted_label"), d.get("relation")
        anchor_label = d.get("anchor")
        if not lab or rel not in ("to the left of", "to the right of",
                                  "above", "below") or not anchor_label:
            return -1, None, "bad counting derivation"
        anchor = find_pinned_instance(anchor_label,
                                      d.get("anchor_attributes") or [], full_objs)
        if anchor is None:
            return -1, None, "counting anchor not uniquely pinnable"
        ax, ay = box_center(anchor["bounding_box"])
        n = 0
        for o in full_objs:
            if o["label"] != lab:
                continue
            cx, cy = box_center(o["bounding_box"])
            if rel == "to the left of" and cx < ax:
                n += 1
            elif rel == "to the right of" and cx > ax:
                n += 1
            elif rel == "above" and cy < ay:
                n += 1
            elif rel == "below" and cy > ay:
                n += 1
        q = (f"How many {pluralize(lab)} are {rel} the "
             f"{attr_phrase(anchor_label, d.get('anchor_attributes') or [])}?")
        return n, q, None
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

        # (b) v2: stored core_nodes are a property of the frozen graph file.
        # (i) every compact edge touches a STORED core node;
        # (ii) stored core_nodes are a subset of the compact labels;
        # (iii) the v2 query's named labels (+ answer label, except
        # counting's integer answer) are all compact labels. For counting
        # only the anchor must be a compact label (the counted label is a
        # whole-image count).
        per_gate_n["b"] += 1
        named = query_named_labels(query)
        stored_core = set(small.get("core_nodes") or [])
        if query.get("template") == "counting":
            required = {query.get("derivation", {}).get("anchor")}
        else:
            required = set(named) | {query.get("answer")}
        b_err = []
        if not stored_core <= comp_labels:
            b_err.append(f"stored core_nodes {sorted(stored_core)} not subset "
                         f"of compact labels {sorted(comp_labels)}")
        untouched = [r for r in comp_rels
                     if r["subject"] not in stored_core and r["object"] not in stored_core]
        if untouched:
            b_err.append(f"{len(untouched)} compact edges touch no stored core node, "
                         f"e.g. {untouched[0]}")
        if not required <= comp_labels:
            b_err.append(f"query labels {sorted(x for x in required if x)} not all "
                         f"compact labels {sorted(comp_labels)}")
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
        orphan = [l for l in comp_labels if l not in stored_core and l not in rel_labels]
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

        # (e/f/i) the single query: re-derive, key edge(s), between geometry
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
            elif query.get("template") == "counting":
                if not isinstance(query.get("answer"), int) or isinstance(query.get("answer"), bool):
                    fail(label, "e", f"counting answer {query.get('answer')!r} is not an integer")
                elif ans != query.get("answer"):
                    fail(label, "e", f"Q {query.get('question')!r}: re-derived count "
                                     f"{ans} != stated answer {query.get('answer')!r}")
                elif ans < 1:
                    fail(label, "e", f"Q {query.get('question')!r}: count {ans} < 1")
                else:
                    per_gate_ok["e"] += 1
                    per_gate_ok["f"] += 1  # counting: no key edge; count re-derived above
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
                elif query.get("template") == "between_attr":
                    per_gate_n["i"] += 1
                    d = query.get("derivation") or {}
                    counts = {}
                    for o in full_objs:
                        counts[o["label"]] = counts.get(o["label"], 0) + 1
                    a_ok = find_pinned_instance(d.get("anchor_a"),
                                                d.get("anchor_a_attributes") or [],
                                                full_objs) is not None
                    b_ok = find_pinned_instance(d.get("anchor_b"),
                                                d.get("anchor_b_attributes") or [],
                                                full_objs) is not None
                    if not a_ok or not b_ok:
                        fail(label, "i", "between_attr anchor not uniquely pinnable")
                    elif counts.get(query.get("answer"), 0) != 1:
                        fail(label, "i", "between_attr answer not single-instance")
                    else:
                        per_gate_ok["i"] += 1
                kes = key_edges(query)
                if not kes:
                    per_gate_ok["f"] += 1  # geometry/count template: re-derived above
                elif all(any((r["subject"], r["relation"], r["object"]) == ke
                             for r in comp_rels) for ke in kes):
                    per_gate_ok["f"] += 1
                else:
                    missing = [ke for ke in kes
                               if not any((r["subject"], r["relation"], r["object"]) == ke
                                          for r in comp_rels)]
                    fail(label, "f", f"key edge(s) {missing} not in compact graph")

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
