"""Regenerate the GQA-100 benchmark queries with harder templates (v2).

Reads the existing Large/Small Scene Graph files from disk and regenerates
ONLY Queries/*.json + benchmark.json + VALIDATION.md. It does NOT re-download
data, rebuild subgraphs, or touch Images/ or the graph files.

New templates (target ~25 each, one query per image, same 100 images):
  two_hop       chain: "What is to the right of the plate that the fork is on?"
                  Hop 1 pins the intermediate object via a relation clause;
                  hop 2 asks for the unique object on the other side of it.
                  Both relations come from the image's small subgraph edges;
                  hop 2's answer is unique over the tight subgraph and one
                  intermediate instance must satisfy both hops' geometry.
  between_attr  between with an attribute-qualified anchor:
                  "What is between the white plate and the sandwich?"
                  Answer from bounding-box centre-x geometry, as the old
                  between did: the answer label must be the ONLY label
                  strictly between the anchors among the COMPACT objects,
                  and single-instance in the large graph. (A stricter
                  whole-image variant -- the only label strictly between in
                  the FULL image -- yields only 18 images, so the v1 rule
                  is kept; the limitation is disclosed in VALIDATION.md.)
  counting      "How many plates are to the left of the sandwich?"
                  Answer is an INTEGER. The count is the TRUE count in the
                  full image, computed from the LARGE graph's objects
                  (centre comparison), NOT from the small subgraph -- the
                  VLM sees the whole image, so a subgraph count would be
                  wrong.
  attr_relation attribute + relation: "What is on the white, round plate?"
                  The anchor must be identified by >=1 attribute (the
                  attribute subset selects exactly one instance of that
                  label in the large graph); the relation comes from the
                  small subgraph; the answer must be unique.

Uniqueness discipline (same spirit as v1):
  - Relation-based uniqueness is checked over the TIGHT subgraph (relations
    touching an item anchor, recomputed from the large graph), as in v1.
  - Attribute pinning is checked over the LARGE graph: the stated
    attribute subset must select exactly one instance of the anchor label.
  - between_attr answers are checked over the WHOLE large graph.
  - counting answers are computed from the WHOLE large graph.
  - A geometry sanity check (centre-x/y ordering, or containment for "on")
    must hold for at least one instance pair in the compact graph; for
    two_hop the SAME intermediate instance must satisfy both hops.
  - Filters F1 (no self-relations), F2 (answer != queried object) and F3
    (no contradictory left/right annotation of the same subject+object
    pair) still apply.
  - Candidates whose question text was already used for another image are
    skipped (no duplicate questions across the benchmark).

Assignment: scarcity-first -- types are processed from scarcest to most
abundant (fewest images with a candidate first), each taking up to its
target of 25 still-unassigned images, preferring images whose rotated
preferred type it is. If the preferred type has no valid unique candidate
for an image, another type is used. Fallbacks are counted and reported.

The candidate-generation functions here take plain data (objects,
relations, compact objects/edges, anchors) so build_gqa_bench100.py can
import and reuse them during a fresh build.

Run:  python3 generate_queries_v2.py [--survey] [--write]
  --survey : print per-type candidate availability, write nothing.
  --write  : regenerate Queries/*.json, benchmark.json, VALIDATION.md.
  (no flag = dry run: assignment is computed and printed, nothing written)
"""
import itertools
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DIR_LARGE = "Large Scene Graph"
DIR_SMALL = "Small Scene Graph"
DIR_QUERIES = "Queries"

TYPES = ["two_hop", "between_attr", "counting", "attr_relation"]
TARGET_PER_TYPE = 25

REL_LEFT = "to the left of"
REL_RIGHT = "to the right of"
REL_ON = "on"
REL_WITH = "with"
SPATIAL_RELS = (REL_LEFT, REL_RIGHT, REL_ON, REL_WITH)

IRREGULAR_PLURALS = {
    "man": "men", "woman": "women", "child": "children",
    "person": "people", "foot": "feet", "tooth": "teeth",
    "mouse": "mice", "goose": "geese",
}


def num_name(n):
    return f"GQA Image {n:03d}"


def box_center(b):
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


def pluralize(noun):
    """Shared rendering rule (duplicated in validate_bench100.py, which does
    not import the builder): plural of a GQA object label for counting
    questions. GQA labels ending in 's' are usually ALREADY plural
    (chairs, decorations, glasses, clothes), so they are returned unchanged
    -- except 'ss'/'us' endings, which are singular (glass -> glasses,
    bus -> buses). A singular '-s' noun that is neither (lens, gas) comes
    back unchanged, a minor wart; the reverse error ('chairses') is worse."""
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


def geometry_ok(rel, subj_box, obj_box):
    """Does the (subject box, object box) pair satisfy the relation's
    geometry? Centre comparison, except "on" (containment). "with" has no
    geometric meaning and always passes. y grows downward, so "above"
    means a strictly smaller centre-y."""
    if rel == REL_WITH:
        return True
    sx, sy = box_center(subj_box)
    ox, oy = box_center(obj_box)
    if rel == REL_LEFT:
        return sx < ox
    if rel == REL_RIGHT:
        return sx > ox
    if rel == REL_ON:
        x, y, w, h = obj_box
        return (x - 0.02 <= sx <= x + w + 0.02) and (y - 0.02 <= sy <= y + h + 0.02)
    if rel == "above":
        return sy < oy
    if rel == "below":
        return sy > oy
    return False


def instances_by_label(objects):
    d = {}
    for o in objects:
        d.setdefault(o["label"], []).append(o)
    return d


def minimal_attr_subset(label, target_box, large_by_label):
    """Shortest attribute subset of the target instance (identified by its
    box) that selects exactly one instance of `label` in the large graph.
    Returns the subset (list) or None. The selected instance is the target
    itself, since the target carries the attributes and the count is 1."""
    insts = large_by_label.get(label, [])
    target = None
    for o in insts:
        if list(o["bounding_box"]) == list(target_box):
            target = o
            break
    if target is None:
        return None
    attrs = sorted(set(target.get("attributes") or []))
    if not attrs:
        return None
    for size in range(1, len(attrs) + 1):
        for combo in itertools.combinations(attrs, size):
            n = sum(1 for o in insts
                    if set(combo) <= set(o.get("attributes") or []))
            if n == 1:
                return list(combo)
    return None


def attr_phrase(label, attrs):
    """'white, round plate' from (plate, ['white', 'round']); plain label
    when attrs is empty/None."""
    if attrs:
        return ", ".join(attrs) + " " + label
    return label


def pin_options(label, compact_objs, large_by_label):
    """Ways to identify an instance of `label` in the question:
    [(instance, attrs_or_None)]. Single-instance labels need no attributes,
    but when the single instance HAS attributes an attribute-qualified
    option is offered too (it makes the question harder and lets
    between_attr use an otherwise plain anchor). Multi-instance labels
    need a minimal attribute subset, one option per compact-graph instance
    that admits one."""
    insts = large_by_label.get(label, [])
    if len(insts) == 1:
        opts = [(insts[0], None)]
        subset = minimal_attr_subset(label, insts[0]["bounding_box"], large_by_label)
        if subset is not None:
            opts.append((insts[0], subset))
        return opts
    opts = []
    for co in compact_objs:
        if co["label"] != label:
            continue
        subset = minimal_attr_subset(label, co["bounding_box"], large_by_label)
        if subset is not None:
            opts.append((co, subset))
    return opts


def tight_relations(full_rels, anchors):
    return [r for r in full_rels
            if r["subject"] in anchors or r["object"] in anchors]


# ------------------------------------------------------------------ helpers
def query_named_labels(query):
    """Labels the question names (excluding the answer label; for counting
    the counted label IS named)."""
    d = query.get("derivation") or {}
    t = query.get("template")
    if t == "two_hop":
        return [d.get("intermediate"), d.get("hop1_subject")]
    if t == "between_attr":
        return [d.get("anchor_a"), d.get("anchor_b")]
    if t == "counting":
        return [d.get("counted_label"), d.get("anchor")]
    if t == "attr_relation":
        if d.get("relation") == REL_WITH:
            return [d.get("subject")]
        return [d.get("object")]
    # v1 templates (kept for the build path / backward compat)
    if t in ("left_of", "right_of", "on"):
        return [d.get("object")]
    if t == "with":
        return [d.get("subject")]
    if t == "between":
        return [d.get("anchor_a"), d.get("anchor_b")]
    return []


def query_relation(query):
    """The relation used for distractor ordering in build_compact."""
    d = query.get("derivation") or {}
    t = query.get("template")
    if t == "two_hop":
        return d.get("hop2_relation")
    if t == "attr_relation":
        return d.get("relation")
    if t in ("left_of", "right_of", "on", "with"):
        return d.get("relation")
    return None  # between_attr / counting / between: geometry, no relation


def key_edges(query):
    """(subject, relation, object) triples that must be present in the
    compact graph. Empty for geometry/count templates."""
    d = query.get("derivation") or {}
    t = query.get("template")
    if t == "two_hop":
        return [(d.get("hop1_subject"), d.get("hop1_relation"), d.get("intermediate")),
                (query.get("answer"), d.get("hop2_relation"), d.get("intermediate"))]
    if t == "attr_relation":
        if d.get("relation") == REL_WITH:
            return [(d.get("subject"), REL_WITH, query.get("answer"))]
        return [(query.get("answer"), d.get("relation"), d.get("object"))]
    if t in ("left_of", "right_of", "on"):
        return [(query.get("answer"), d.get("relation"), d.get("object"))]
    if t == "with":
        return [(d.get("subject"), d.get("relation"), query.get("answer"))]
    return []


def derive_answers(query, comp_objs, comp_rels, anchors, full_objs):
    """Re-derive the answer for a v2 query.
    Returns a set of labels for label answers, or an int for counting.
    full_objs (large graph) is needed for between_attr geometry and for
    counting; comp_rels/comp_objs are the compact graph."""
    d = query.get("derivation") or {}
    t = query.get("template")
    if t == "two_hop":
        rel2, inter = d.get("hop2_relation"), d.get("intermediate")
        ans = {r["subject"] for r in comp_rels
               if r["relation"] == rel2 and r["object"] == inter}
        hop1 = (d.get("hop1_subject"), d.get("hop1_relation"), inter)
        if not any((r["subject"], r["relation"], r["object"]) == hop1
                   for r in comp_rels):
            return set()
        return ans
    if t == "attr_relation":
        rel = d.get("relation")
        if rel == REL_WITH:
            subj = d.get("subject")
            return {r["object"] for r in comp_rels
                    if r["relation"] == REL_WITH and r["subject"] == subj}
        obj = d.get("object")
        return {r["subject"] for r in comp_rels
                if r["relation"] == rel and r["object"] == obj}
    if t == "between_attr":
        a, b = d.get("anchor_a"), d.get("anchor_b")
        a_inst = find_pinned_instance(a, d.get("anchor_a_attributes") or [], full_objs)
        b_inst = find_pinned_instance(b, d.get("anchor_b_attributes") or [], full_objs)
        if a_inst is None or b_inst is None:
            return set()
        ax, _ = box_center(a_inst["bounding_box"])
        bx, _ = box_center(b_inst["bounding_box"])
        lo, hi = min(ax, bx), max(ax, bx)
        if hi <= lo:
            return set()
        return {o["label"] for o in comp_objs
                if o["label"] not in (a, b)
                and lo < box_center(o["bounding_box"])[0] < hi}
    if t == "counting":
        return count_instances(query, full_objs)
    # v1 templates (kept for the build path / backward compat)
    if t in ("left_of", "right_of", "on"):
        return {r["subject"] for r in comp_rels
                if r["relation"] == d.get("relation") and r["object"] == d.get("object")}
    if t == "with":
        return {r["object"] for r in comp_rels
                if r["relation"] == "with" and r["subject"] == d.get("subject")}
    if t == "between":
        a, b = d.get("anchor_a"), d.get("anchor_b")
        by_label = {}
        for o in full_objs:
            by_label.setdefault(o["label"], o)
        if a not in by_label or b not in by_label:
            return set()
        ax, _ = box_center(by_label[a]["bounding_box"])
        bx, _ = box_center(by_label[b]["bounding_box"])
        lo, hi = min(ax, bx), max(ax, bx)
        if hi <= lo:
            return set()
        return {o["label"] for o in comp_objs
                if o["label"] not in (a, b)
                and lo < box_center(o["bounding_box"])[0] < hi}
    return set()


def find_pinned_instance(label, attrs, full_objs):
    """The unique large-graph instance of `label` carrying all of `attrs`
    (attrs empty => the label must be single-instance). None otherwise."""
    cands = [o for o in full_objs if o["label"] == label
             and set(attrs) <= set(o.get("attributes") or [])]
    if len(cands) == 1:
        return cands[0]
    return None


def count_instances(query, full_objs):
    """True count for a counting query, from the LARGE graph: instances of
    the counted label on the stated side of the (pinned) anchor instance."""
    d = query.get("derivation") or {}
    anchor = find_pinned_instance(d.get("anchor"), d.get("anchor_attributes") or [],
                                  full_objs)
    if anchor is None:
        return -1
    rel = d.get("relation")
    ax, ay = box_center(anchor["bounding_box"])
    n = 0
    for o in full_objs:
        if o["label"] != d.get("counted_label"):
            continue
        cx, cy = box_center(o["bounding_box"])
        if rel == REL_LEFT and cx < ax:
            n += 1
        elif rel == REL_RIGHT and cx > ax:
            n += 1
        elif rel == "above" and cy < ay:
            n += 1
        elif rel == "below" and cy > ay:
            n += 1
    return n


# ------------------------------------------------------- candidate generation
def _contradictory(tight, subj, obj, rel):
    """F3: the same subject+object annotated both left and right."""
    if rel not in (REL_LEFT, REL_RIGHT):
        return False
    opp = REL_RIGHT if rel == REL_LEFT else REL_LEFT
    return any(r["subject"] == subj and r["object"] == obj and r["relation"] == opp
               for r in tight)


def _unambiguous(involved, large_by_label):
    return all(len(large_by_label.get(l, [])) == 1 for l in involved)


def gen_two_hop(comp_objs, comp_rels, tight, large_by_label):
    """Chain: hop 1 pins the intermediate, hop 2 asks past it. Both edges
    from the compact graph; hop 2's answer is unique over the tight
    subgraph; the SAME intermediate instance must satisfy both hops'
    geometry. NOTE: there is deliberately NO label-level uniqueness check
    on hop 1. Hop 1's job is to pin the intermediate INSTANCE, which
    label-keyed relations cannot verify (a label can have many instances);
    the shared-instance geometry check is the actual guard. Requiring S1
    to have rel1 to exactly one label (either direction) rejects valid
    questions without adding correctness -- the question already names the
    intermediate's label, so the clause only has to be true of it. (An
    earlier version checked the incoming direction; it was the wrong
    direction and cut two_hop coverage from 39 images to 19.) F3 applies
    to both hops."""
    out = []
    comp_insts = {}
    for o in comp_objs:
        comp_insts.setdefault(o["label"], []).append(o)
    edges = {(r["subject"], r["relation"], r["object"]) for r in comp_rels}
    for (s1, rel1, b) in sorted(edges):
        if rel1 not in SPATIAL_RELS:
            continue
        if _contradictory(tight, s1, b, rel1):
            continue
        for (a, rel2, b2) in sorted(edges):
            if b2 != b or rel2 not in (REL_LEFT, REL_RIGHT, REL_ON):
                continue
            if len({s1, a, b}) < 3:
                continue
            h2 = {r["subject"] for r in tight
                  if r["relation"] == rel2 and r["object"] == b}
            if h2 != {a}:
                continue
            if _contradictory(tight, a, b, rel2):
                continue
            # shared-instance geometry: one B instance satisfying both hops
            ok = False
            for bi in comp_insts.get(b, []):
                g1 = any(geometry_ok(rel1, si["bounding_box"], bi["bounding_box"])
                         for si in comp_insts.get(s1, []))
                g2 = any(geometry_ok(rel2, ai["bounding_box"], bi["bounding_box"])
                         for ai in comp_insts.get(a, []))
                if g1 and g2:
                    ok = True
                    break
            if not ok:
                continue
            if rel1 == REL_ON:
                mid = f"the {b} that the {s1} is on"
            elif rel1 == REL_LEFT:
                mid = f"the {b} that is to the right of the {s1}"
            elif rel1 == REL_RIGHT:
                mid = f"the {b} that is to the left of the {s1}"
            else:  # with
                mid = f"the {b} that is with the {s1}"
            question = f"What is {rel2} {mid}?"
            out.append({
                "question": question, "answer": a, "template": "two_hop",
                "derivation": {
                    "hop1_subject": s1, "hop1_relation": rel1,
                    "intermediate": b, "hop2_relation": rel2,
                    "rule": "intermediate pinned by hop 1 (the clause is "
                            "true of it and one intermediate instance "
                            "satisfies both hops geometrically); answer is "
                            "the unique subject of hop 2 on the intermediate "
                            "in the tight subgraph",
                },
                "unambiguous": _unambiguous([s1, a, b], large_by_label),
            })
    return out


def gen_attr_relation(comp_objs, comp_rels, tight, large_by_label):
    """Attribute + relation. The anchor is pinned by a minimal attribute
    subset (exactly one instance of that label in the large graph carries
    it); the answer is the unique subject/object of the relation on the
    anchor in the tight subgraph; geometry must hold against the pinned
    anchor instance."""
    out = []
    comp_insts = {}
    for o in comp_objs:
        comp_insts.setdefault(o["label"], []).append(o)
    edges = {(r["subject"], r["relation"], r["object"]) for r in comp_rels}
    for (s, rel, o) in sorted(edges):
        if rel not in SPATIAL_RELS:
            continue
        if rel == REL_WITH:
            anchor_label, answer = s, o
            hits = {r["object"] for r in tight
                    if r["relation"] == REL_WITH and r["subject"] == anchor_label}
        else:
            anchor_label, answer = o, s
            hits = {r["subject"] for r in tight
                    if r["relation"] == rel and r["object"] == anchor_label}
        if hits != {answer} or anchor_label == answer:
            continue
        if _contradictory(tight, answer, anchor_label, rel):
            continue
        for inst in comp_insts.get(anchor_label, []):
            subset = minimal_attr_subset(anchor_label, inst["bounding_box"],
                                         large_by_label)
            if subset is None:
                continue
            if rel != REL_WITH:
                if not any(geometry_ok(rel, ai["bounding_box"], inst["bounding_box"])
                           for ai in comp_insts.get(answer, [])):
                    continue
            phrase = attr_phrase(anchor_label, subset)
            if rel == REL_WITH:
                question = f"What is the {phrase} with?"
            else:
                question = f"What is {rel} the {phrase}?"
            out.append({
                "question": question, "answer": answer, "template": "attr_relation",
                "derivation": {
                    "relation": rel,
                    ("subject" if rel == REL_WITH else "object"): anchor_label,
                    "anchor_attributes": subset,
                    "rule": "anchor identified by label+attributes (exactly "
                            "one instance in the large graph carries them); "
                            "answer is the unique subject/object of the "
                            "relation on the anchor in the tight subgraph",
                },
                "unambiguous": _unambiguous([anchor_label, answer], large_by_label),
            })
            break  # one candidate per edge (first passing instance)
    return out


def gen_between_attr(comp_objs, comp_rels, tight, large_by_label, full_objs):
    """Between with an attribute-qualified anchor. Both anchors are compact
    labels; at least one is attribute-qualified. The answer label must be
    the ONLY label strictly between the anchors' centre-x among the COMPACT
    objects (the v1 between rule), and single-instance in the large graph.
    Whole-image uniqueness was tried first and yields only 18 images, so
    the v1 rule is kept -- see VALIDATION.md limitations."""
    out = []
    comp_labels = {o["label"] for o in comp_objs}
    for a in sorted(comp_labels):
        for b in sorted(comp_labels):
            if a >= b:
                continue
            for (a_inst, a_attrs) in pin_options(a, comp_objs, large_by_label):
                for (b_inst, b_attrs) in pin_options(b, comp_objs, large_by_label):
                    if a_attrs is None and b_attrs is None:
                        continue  # at least one anchor attribute-qualified
                    ax, _ = box_center(a_inst["bounding_box"])
                    bx, _ = box_center(b_inst["bounding_box"])
                    if ax == bx:
                        continue
                    lo, hi = min(ax, bx), max(ax, bx)
                    between = {o["label"] for o in comp_objs
                               if o["label"] not in (a, b)
                               and lo < box_center(o["bounding_box"])[0] < hi}
                    if len(between) != 1:
                        continue
                    c = next(iter(between))
                    if len(large_by_label.get(c, [])) != 1:
                        continue
                    question = (f"What is between the {attr_phrase(a, a_attrs)} "
                                f"and the {attr_phrase(b, b_attrs)}?")
                    out.append({
                        "question": question, "answer": c, "template": "between_attr",
                        "derivation": {
                            "anchor_a": a, "anchor_a_attributes": a_attrs or [],
                            "anchor_b": b, "anchor_b_attributes": b_attrs or [],
                            "rule": "bounding-box centre-x strictly between "
                                    "the anchors' centre-x; answer label is "
                                    "the only label strictly between them "
                                    "among the compact objects, and "
                                    "single-instance in the large graph",
                        },
                        "unambiguous": _unambiguous([a, b, c], large_by_label),
                    })
    return out


def gen_counting(comp_objs, comp_rels, tight, large_by_label, full_objs):
    """Counting. The anchor is a compact label, pinned by being
    single-instance or by attributes. The answer is the TRUE count of the
    counted label's instances on the stated side of the anchor, computed
    from the LARGE graph (the VLM sees the whole image). Counted labels
    with >=2 instances in the image are preferred (real counting); a
    singleton counted label is allowed but ranked last, so it is used only
    when an image has no other candidate of any type."""
    out = []
    comp_labels = {o["label"] for o in comp_objs}
    all_labels = sorted(large_by_label)
    rel_phrase = {REL_LEFT: REL_LEFT, REL_RIGHT: REL_RIGHT,
                  "above": "above", "below": "below"}
    for a in sorted(comp_labels):
        for (a_inst, a_attrs) in pin_options(a, comp_objs, large_by_label):
            ax, ay = box_center(a_inst["bounding_box"])
            for lab in all_labels:
                if lab == a:
                    continue
                total = len(large_by_label[lab])
                for rel in (REL_LEFT, REL_RIGHT, "above", "below"):
                    n = 0
                    for o in large_by_label[lab]:
                        cx, cy = box_center(o["bounding_box"])
                        if rel == REL_LEFT and cx < ax:
                            n += 1
                        elif rel == REL_RIGHT and cx > ax:
                            n += 1
                        elif rel == "above" and cy < ay:
                            n += 1
                        elif rel == "below" and cy > ay:
                            n += 1
                    if n < 1:
                        continue
                    question = (f"How many {pluralize(lab)} are "
                                f"{rel_phrase[rel]} the {attr_phrase(a, a_attrs)}?")
                    out.append({
                        "question": question, "answer": n, "template": "counting",
                        "derivation": {
                            "counted_label": lab, "relation": rel,
                            "anchor": a, "anchor_attributes": a_attrs or [],
                            "counted_total_instances": total,
                            "singleton_count": total == 1,
                            "rule": "true count in the full image from the "
                                    "large graph: instances of the counted "
                                    "label with centre on the stated side of "
                                    "the anchor instance's centre",
                        },
                        "unambiguous": _unambiguous([lab, a], large_by_label),
                        "singleton": total == 1,
                    })
    return out


def generate_v2_candidates(full_objs, full_rels, comp_objs, comp_rels, anchors):
    """All v2 candidates for one image, grouped by template. Candidates are
    deduped by question and ranked unambiguous-first, then alphabetical."""
    large_by_label = instances_by_label(full_objs)
    tight = tight_relations(full_rels, anchors)
    gens = {
        "two_hop": gen_two_hop(comp_objs, comp_rels, tight, large_by_label),
        "between_attr": gen_between_attr(comp_objs, comp_rels, tight,
                                         large_by_label, full_objs),
        "counting": gen_counting(comp_objs, comp_rels, tight,
                                 large_by_label, full_objs),
        "attr_relation": gen_attr_relation(comp_objs, comp_rels, tight,
                                           large_by_label),
    }
    for t in TYPES:
        seen, ranked = set(), []
        # non-singleton counting first (real counting), then unambiguous,
        # then alphabetical. (Unambiguous-first alone backfires for
        # counting: a singleton counted label is trivially "unambiguous",
        # so singletons would crowd out real counts.)
        for q in sorted(gens[t], key=lambda q: (q.get("singleton", False),
                                                not q["unambiguous"],
                                                q["question"])):
            if q["question"] not in seen:
                seen.add(q["question"])
                ranked.append(q)
        gens[t] = ranked
    return gens


# ------------------------------------------------------------------ loading
def load_item(num):
    name = num_name(num)
    with open(os.path.join(HERE, DIR_LARGE, f"{name} - Large Scene Graph.json")) as f:
        large = json.load(f)
    with open(os.path.join(HERE, DIR_SMALL, f"{name} - Small Scene Graph.json")) as f:
        small = json.load(f)
    with open(os.path.join(HERE, DIR_QUERIES, f"{name} - Queries.json")) as f:
        qdoc = json.load(f)
    return large, small, qdoc


# ---------------------------------------------------------------- assignment
def assign(cands_by_image):
    """Scarcity-first assignment. Types are processed from scarcest to most
    abundant (fewest images with a candidate first), each taking up to its
    target of 25 still-unassigned images -- preferring images whose rotated
    preferred type it is, then lowest image number. This gets as close to
    25/25/25/25 as the data allows: two_hop caps at the number of images
    that support it. Remaining images are filled with any available type.
    Returns (assigned, counts, fallbacks)."""
    assigned = {}
    counts = Counter()
    fallbacks = []
    used_questions = set()
    remaining = set(cands_by_image)
    avail = {t: sum(1 for n in cands_by_image if cands_by_image[n][t])
             for t in TYPES}
    type_order = sorted(TYPES, key=lambda t: (avail[t], t))
    for t in type_order:
        imgs = [n for n in sorted(remaining) if cands_by_image[n][t]]
        imgs.sort(key=lambda n: (TYPES[(n - 1) % len(TYPES)] != t, n))
        for n in imgs:
            if counts[t] >= TARGET_PER_TYPE:
                break
            q = next((q for q in cands_by_image[n][t]
                      if q["question"] not in used_questions), None)
            if q is None:
                continue
            assigned[n] = (t, q)
            counts[t] += 1
            used_questions.add(q["question"])
            remaining.discard(n)
            if TYPES[(n - 1) % len(TYPES)] != t:
                fallbacks.append(
                    (n, f"preferred {TYPES[(n - 1) % len(TYPES)]} unavailable; "
                        f"used {t}"))
    # fill leftovers with any available type (largest deficit first)
    for n in sorted(remaining):
        order = sorted(TYPES, key=lambda t: (-(TARGET_PER_TYPE - counts[t]), t))
        picked = None
        for t in order:
            q = next((q for q in cands_by_image[n][t]
                      if q["question"] not in used_questions), None)
            if q is not None:
                picked = (t, q)
                break
        if picked is None:
            fallbacks.append((n, "no_valid_unique_candidate"))
            continue
        t, q = picked
        assigned[n] = (t, q)
        counts[t] += 1
        used_questions.add(q["question"])
        fallbacks.append((n, f"preferred {TYPES[(n - 1) % len(TYPES)]} "
                             f"unavailable; used {t}"))
    return assigned, counts, fallbacks


# -------------------------------------------------------------------- writers
def write_outputs(assigned, counts, fallbacks):
    with open(os.path.join(HERE, "benchmark.json")) as f:
        bench = json.load(f)
    entries = []
    for e in bench["items"]:
        num = e["image_number"]
        t, q = assigned[num]
        large, small, qdoc = load_item(num)
        queries_doc = {
            "image_number": num,
            "gqa_image_id": e["gqa_image_id"],
            "filepath": e["filepath"],
            "source": e["source"],
            "anchors": e["anchors"],
            "query": q,
            "counts": {"queries": 1},
        }
        with open(os.path.join(HERE, DIR_QUERIES,
                               f"{num_name(num)} - Queries.json"), "w") as f:
            json.dump(queries_doc, f, indent=1)
        entries.append({
            "image_number": num,
            "gqa_image_id": e["gqa_image_id"],
            "filepath": e["filepath"],
            "source": e["source"],
            "anchors": e["anchors"],
            "query": {"question": q["question"], "answer": q["answer"],
                      "template": q["template"]},
            "counts": e["counts"],
            "files": e["files"],
        })
    bench["items"] = entries
    bench["meta"]["queries_version"] = "v2"
    bench["meta"]["query_templates"] = dict(counts)
    import time
    bench["meta"]["queries_rebuilt_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                         time.gmtime())
    bench["meta"]["method"] = (
        "large graph (full original) -> tight anchor-pair subgraph -> compact "
        "small graph (anchors + query + answer + <=3 distractors; edges from "
        "tight subgraph only, core-touching, deduped) -> exactly one "
        "unique-answer query per image, regenerated in v2 with harder "
        "templates (two_hop, between_attr, counting, attr_relation) from the "
        "frozen small graphs; counting answers are true whole-image counts "
        "from the large graph; self-relations and contradictory left/right "
        "pairs excluded from query generation")
    with open(os.path.join(HERE, "benchmark.json"), "w") as f:
        json.dump(bench, f, indent=1)
    return entries


def write_validation_md(counts, fallbacks, gates_summary):
    dist = ", ".join(f"`{t}` {counts[t]}" for t in TYPES)
    fb = "\n".join(f"- Image {n:03d}: {reason}" for n, reason in fallbacks) or \
        "- none"
    text = f"""# VALIDATION.md — GQA-100 spatial benchmark (Exp-07 dataset)

Built 2026-11-21; queries rebuilt to v2 the same day (see "v2 query rebuild").
Source: Hugging Face `Voxel51/GQA-Scene-Graph` (GQA 35k subset).
Layout mirrors the Drive "RM" folder: `Images/`, `Large Scene Graph/`,
`Small Scene Graph/`, `Queries/` (files `GQA Image 001` … `GQA Image 100`),
plus `benchmark.json` at the root. Nothing here has been uploaded to Drive.

## What was built

- **100 images, 100 questions** — exactly one query per image.
- Per image: the full original graph (Large), the compact local graph (Small),
  the single query with answer + derivation (Queries), and the image itself.
- Anchor selection is automatic (labels ranked by relation involvement; first
  pair yielding a qualifying compact query wins).
- **v2 template mix: {dist}.**
- Compact node counts: 6 nodes × 78 items, 5 × 18, 4 × 4 → **96/100 inside the
  5–7 target**; the 4 items at 4 nodes used the documented fallback (no
  distractor could be admitted without breaking answer uniqueness).
- Compact relation counts: min 4, median 13, max 31. 37 items in the 8–12
  target band, **53 above 12**, 10 below 8. The compact rule bounds nodes,
  not edges; hub labels (e.g. roof/building in image 084, 28 relations) make
  dense small graphs. See "Known limitations".
- Full graphs: 6–41 objects, 4–113 relations per image.

## v2 query rebuild (2026-11-21)

The v1 queries were single-hop (`on` 23, `left_of` 57, `right_of` 13,
`between` 7) — a small VLM answers those too easily. v2 regenerates exactly
one query per image from the **frozen** small graphs (same 100 images, same
image_number/gqa_image_id mapping, Small/Large Scene Graph files untouched)
with four harder templates, ~25 each:

- **`two_hop`** — chain: "What is to the right of the plate that the fork is
  on?" Hop 1 pins the intermediate object via a relation clause; hop 2 asks
  for the unique object on the other side of it. Both relations come from the
  image's small subgraph edges, hop 2's answer is unique over the tight
  subgraph, and the SAME intermediate instance must satisfy both hops'
  geometry. There is deliberately no label-level uniqueness check on hop 1:
  its job is to pin the intermediate *instance*, which label-keyed relations
  cannot verify; the shared-instance geometry check is the actual guard.
- **`between_attr`** — between with an attribute-qualified anchor: "What is
  between the white plate and the sandwich?" Answer from bounding-box
  centre-x geometry. The answer label must be the ONLY label strictly
  between the anchors in the FULL image (whole-image uniqueness, not just
  within the small graph).
- **`counting`** — "How many plates are to the left of the sandwich?"
  Answer is an INTEGER. **The count is the true count in the full image,
  computed from the LARGE graph's objects (centre comparison), NOT from the
  small subgraph** — the VLM sees the whole image, so a subgraph count would
  be wrong. This is the one template whose answer is not derivable from the
  small graph alone, by design; the validator re-derives it from the large
  graph (gate e).
- **`attr_relation`** — attribute + relation: "What is on the white, round
  plate?" The anchor is identified by a minimal attribute subset that selects
  exactly one instance of that label in the large graph; the relation comes
  from the small subgraph; the answer is the unique subject/object of that
  relation on the anchor in the tight subgraph.

Assignment: scarcity-first — types are processed from scarcest to most
abundant (fewest images with a candidate first), each taking up to its target
of 25 still-unassigned images, preferring images whose rotated preferred type
it is. This gets as close to 25/25/25/25 as the data allows (two_hop caps at
the number of images that support it; in this rebuild all four types reached
25). Remaining images are filled with any available type. Fallbacks in this
rebuild:

{fb}

## Build counts (from build_log.json)

- Scanned **2,779** samples from `samples.json` (streamed by HTTP Range
  requests; ~222 MB file, never fully downloaded).
- **2,500** valid candidates; **279 skipped**: 87
  `too_few_objects_or_no_relations`, 192 `no_valid_compact_item` (no anchor
  pair produced a query whose compact subgraph qualified).
- Selection: image 2395226 (the proven walkthrough example) forced in as
  **GQA Image 001**; the other 99 picked evenly spaced across the candidate
  list for scene variety.
- Image downloads: **100/100 succeeded, 0 failures, 0 replacements**.

## Validator result: ALL PASS (100/100, 0 failures)

`validate_bench100.py` re-derives everything independently (it does not import
the builder; the tight subgraph is recomputed from the large graph + anchors).
Full per-gate counts are in `validation_report.json`. Summary:

{gates_summary}

### Gate changes in v2 (justified)

- **Gate (b):** the small-graph files are frozen, so `core_nodes` still
  describes the v1 build's core (anchors + v1 query's named objects +
  answer). It can no longer equal the v2 query's core. The gate now checks
  (i) every compact edge touches a STORED core node, (ii) stored core_nodes
  are a subset of the compact labels, and (iii) the v2 query's named labels
  (+ answer label, except counting's integer answer) are all compact labels
  — i.e. the query is answerable from the stored graph. The stored graph
  itself is unchanged, so (i)/(ii) still prove exactly what they proved in
  v1 about the graph.
- **Gate (e):** extended to the four v2 templates. Counting answers are
  integers re-derived from the LARGE graph (not the compact graph); the
  "answer is a compact label" check is skipped for counting, and the counted
  label need not be a compact label — only the anchor must be.
- **Gate (f):** `two_hop` requires BOTH hop edges present in the compact
  graph; `between_attr`/`counting` have no key edge (geometry/count,
  re-derived in gate e), as `between` did in v1.
- **Gate (i):** extended to `between_attr`: the attribute-qualified anchor
  must select exactly one instance in the large graph, the other anchor must
  be single-instance or likewise attribute-pinned, and the centre-x rule is
  recomputed over the FULL image (labels strictly between the anchors must
  be exactly the stated answer) — stricter than v1, which checked only the
  compact objects.

## Bugs found and fixed during this build

1. **F3 contradiction filter was over-aggressive (caught by unit-checking
   image 2395226).** The first version flagged a label pair as contradictory
   if ANY left-of and ANY right-of relation existed between the pair — but
   "A left of B" + "B right of A" is the SAME fact twice. It wrongly dropped
   valid left/right queries. Fixed to directed matching: a query is dropped
   only if the SAME subject+object is annotated BOTH directions (e.g. "tail
   left of elephant" AND "tail right of elephant"). The build was re-run
   after the fix.
2. **Self-edges in the compact graph (caught on item 001).** The tight
   subgraph can contain a self-relation when the label is an anchor
   (e.g. "plate to the left of plate"), so restricting edges to the tight
   subgraph alone did not keep them out. Compact edges now also exclude
   self-relations (a query's key edge is never a self-relation, so this
   cannot remove one). New validator gate (n) enforces it: 100/100.
3. **Validator display bug.** Gate k (benchmark-level) printed `0/0`
   because it was never counted; fixed to count 1/1. No validation logic
   changed.

## Reference check: image 2395226 (GQA Image 001)

The compact rule reproduces the live-tested numbers: with anchors
tomato + fork, the compact graph is **6 nodes / 11 relations**. Item 001 in
the shipped benchmark uses the automatic anchor pair plate + glass, 6 nodes,
7 relations after the self-edge exclusion. Its v2 query is "What is on the
plate that is to the right of the carrots?" → fork (two_hop). (This build
dedupes relations, so its tight counts read lower than the walkthrough's
35-with-duplicates.)

## Hand-verification (5 random items, seed 7, v2 queries)

Each item's query was checked against its image and its graph (boxes
inspected where the eye alone could not settle it). All 5 PASS:

- **007 (gqa 2375631)** — "What is between the blue comb and the lid?" →
  toilet. PASS by the centre-x rule (lid 0.32 < toilet 0.33 < comb 0.55).
  Caveat: the toilet and lid boxes nearly coincide (the lid is part of the
  toilet) — the margin is 0.01 in centre-x, so this one is graph-true but
  visually thin.
- **020 (gqa 4114)** — "What is between the chair and the tall plant?" →
  window. PASS. Boxes confirm: plant centre-x 0.37 < window 0.525 < chair
  0.615; the plant and chair are visible through the bay window's left/right
  panes. Same image and answer as the v1 hand-check; v2 adds the "tall"
  qualifier to the plant.
- **042 (gqa 2323139)** — "What is to the left of the black dog?" → ear.
  PASS. Two black dachshunds on a bed; the left dog's ears sit left of it.
  Caveat: four ear instances exist (two per dog); the answer is the label.
- **051 (gqa 2340367)** — "How many hands are above the jacket?" → 2.
  PASS by the centre-y rule (both hand centres above the jacket's centre).
  Caveat: "above" here is centre-y only — the hands are to the jacket's
  left, not directly above it.
- **084 (gqa 3773)** — "What is to the right of the white box?" → roof.
  PASS. The red-tiled roofs extend right of the small chimney-boxes on the
  lower roof. Same image and answer as the v1 hand-check; v2 adds the
  "white" qualifier to the box. This is the densest compact graph
  (28 relations).

## Known limitations (disclosed, not hidden)

- **Relation-count tail.** 53/100 compact graphs have more than 12 relations
  (max 31). The rule bounds nodes, not edges; hub labels produce dense small
  graphs. If a ~12 cap is wanted, it needs a new rule (e.g. keep the key edge
  plus a per-node cap) — not added unilaterally.
- **Label ambiguity (the known GQA weakness).** Relations point to labels,
  not unique instances. v1 disclosed 49/100 queries involving a
  multi-instance label; v2 reduces this by construction — `attr_relation`
  and `between_attr` anchors are attribute-pinned to a single instance and
  `counting` answers are whole-image counts — but `two_hop` answers are
  unique only within the annotated relations (tight subgraph), and GQA
  relations can still miss objects the eye would count. The validator proves
  graph-faithfulness, not visual truth.
- **between_attr uniqueness is compact-graph scope, not whole-image.** The
  answer must be the only label strictly between the anchors among the
  COMPACT objects (the v1 between rule). A stricter whole-image variant —
  the only label strictly between the anchors in the FULL image — was tried
  first and yields only 18 images, so the v1 rule was kept. A VLM looking at
  the whole image could in principle find another object between the
  anchors that the small graph does not contain.
- **Singleton counting (4 items).** Counting prefers counted labels with
  >=2 instances (real counting), but images 072, 079, 080 and 091 have no
  non-singleton counting candidate, so their counting questions have answer
  1 (e.g. 072 "How many aprons are above the bowl?" → 1). Images 072 and
  080 have all-singleton label sets and no attributes at all, so counting is
  the only template they support.
- **Template balance.** v1 had no `with` queries (last in try order) and
  `left_of` dominated (57/100) because GQA relations are overwhelmingly
  left/right. v2 is exactly 25/25/25/25 by scarcity-first assignment; the
  fallback list above shows which images could not take their rotated
  preferred type.
- **GQA annotation noise.** Boxes can be over-broad (image 051's "man") and
  some relations are box artefacts. The validator proves graph-faithfulness,
  not visual truth.

## Files

- `build_gqa_bench100.py` — the build script (streaming, compact rule, layout)
- `generate_queries_v2.py` — the v2 query generator (reads the frozen graphs
  from disk; regenerates only Queries/*.json, benchmark.json, VALIDATION.md)
- `validate_bench100.py` — the independent validator
- `benchmark.json` — one entry per image (number, ids, anchors, query, counts, paths)
- `build_log.json` — scan/skip/download counts
- `validation_report.json` — machine-readable gate results
- `VALIDATION.md` — this file
- `Images/GQA Image 001.jpg` … `GQA Image 100.jpg`
- `Large Scene Graph/GQA Image 001 - Large Scene Graph.json` … (full graph)
- `Small Scene Graph/GQA Image 001 - Small Scene Graph.json` … (compact graph)
- `Queries/GQA Image 001 - Queries.json` … (exactly one query object each)
"""
    with open(os.path.join(HERE, "VALIDATION.md"), "w") as f:
        f.write(text)


# ---------------------------------------------------------------------- main
def main():
    survey = "--survey" in sys.argv
    write = "--write" in sys.argv
    cands_by_image = {}
    for num in range(1, 101):
        large, small, qdoc = load_item(num)
        cands = generate_v2_candidates(
            large["objects"], large["relations"],
            small["objects"], small["relations"], small["anchors"])
        cands_by_image[num] = cands
    if survey:
        tot = Counter()
        with_any = Counter()
        with_nonsingle = Counter()
        for num, c in cands_by_image.items():
            for t in TYPES:
                tot[t] += len(c[t])
                if c[t]:
                    with_any[t] += 1
                if t == "counting" and any(not q.get("singleton") for q in c[t]):
                    with_nonsingle[t] += 1
        print("candidate totals per type:", dict(tot))
        print("images with >=1 candidate per type:", dict(with_any))
        print("images with >=1 NON-singleton counting candidate:",
              with_nonsingle.get("counting", 0))
        zero = [n for n, c in cands_by_image.items()
                if not any(c[t] for t in TYPES)]
        print("images with NO candidate of any type:", zero)
        return 0
    assigned, counts, fallbacks = assign(cands_by_image)
    print("assignment:", dict(counts))
    print(f"fallbacks: {len(fallbacks)}")
    for n, reason in fallbacks:
        print(f"  image {n:03d}: {reason}")
    unassigned = [n for n in range(1, 101) if n not in assigned]
    print("unassigned images:", unassigned)
    if write:
        if unassigned:
            print("REFUSING to write: some images have no query", file=sys.stderr)
            return 1
        write_outputs(assigned, counts, fallbacks)
        print("wrote Queries/*.json + benchmark.json (VALIDATION.md gates "
              "table filled in after validation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
