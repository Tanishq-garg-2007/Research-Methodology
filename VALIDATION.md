# VALIDATION.md — GQA-100 spatial benchmark (Exp-07 dataset)

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
- **v2 template mix: `two_hop` 25, `between_attr` 25, `counting` 25, `attr_relation` 25.**
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

- Image 006: preferred between_attr unavailable; used two_hop
- Image 010: preferred between_attr unavailable; used two_hop
- Image 016: preferred attr_relation unavailable; used two_hop
- Image 018: preferred between_attr unavailable; used two_hop
- Image 019: preferred counting unavailable; used two_hop
- Image 022: preferred between_attr unavailable; used two_hop
- Image 027: preferred counting unavailable; used two_hop
- Image 030: preferred between_attr unavailable; used two_hop
- Image 031: preferred counting unavailable; used two_hop
- Image 034: preferred between_attr unavailable; used two_hop
- Image 038: preferred between_attr unavailable; used two_hop
- Image 047: preferred counting unavailable; used two_hop
- Image 054: preferred between_attr unavailable; used two_hop
- Image 055: preferred counting unavailable; used two_hop
- Image 004: preferred attr_relation unavailable; used between_attr
- Image 007: preferred counting unavailable; used between_attr
- Image 008: preferred attr_relation unavailable; used between_attr
- Image 009: preferred two_hop unavailable; used between_attr
- Image 011: preferred counting unavailable; used between_attr
- Image 015: preferred counting unavailable; used between_attr
- Image 020: preferred attr_relation unavailable; used between_attr
- Image 021: preferred two_hop unavailable; used between_attr
- Image 023: preferred counting unavailable; used between_attr
- Image 024: preferred attr_relation unavailable; used between_attr
- Image 032: preferred attr_relation unavailable; used between_attr
- Image 035: preferred counting unavailable; used between_attr
- Image 036: preferred attr_relation unavailable; used between_attr
- Image 039: preferred counting unavailable; used between_attr
- Image 003: preferred counting unavailable; used attr_relation
- Image 025: preferred two_hop unavailable; used attr_relation
- Image 041: preferred two_hop unavailable; used attr_relation
- Image 042: preferred between_attr unavailable; used attr_relation
- Image 043: preferred counting unavailable; used attr_relation
- Image 049: preferred two_hop unavailable; used attr_relation
- Image 053: preferred two_hop unavailable; used attr_relation
- Image 057: preferred two_hop unavailable; used attr_relation
- Image 061: preferred two_hop unavailable; used attr_relation
- Image 063: preferred counting unavailable; used attr_relation
- Image 067: preferred counting unavailable; used attr_relation
- Image 071: preferred counting unavailable; used attr_relation
- Image 074: preferred between_attr unavailable; used attr_relation
- Image 075: preferred counting unavailable; used attr_relation
- Image 077: preferred two_hop unavailable; used attr_relation
- Image 078: preferred between_attr unavailable; used attr_relation
- Image 002: preferred between_attr unavailable; used counting
- Image 012: preferred attr_relation unavailable; used counting
- Image 028: preferred attr_relation unavailable; used counting
- Image 033: preferred two_hop unavailable; used counting
- Image 048: preferred attr_relation unavailable; used counting
- Image 052: preferred attr_relation unavailable; used counting
- Image 065: preferred two_hop unavailable; used counting
- Image 068: preferred attr_relation unavailable; used counting
- Image 069: preferred two_hop unavailable; used counting
- Image 072: preferred attr_relation unavailable; used counting
- Image 080: preferred attr_relation unavailable; used counting
- Image 085: preferred two_hop unavailable; used counting
- Image 088: preferred attr_relation unavailable; used counting
- Image 089: preferred two_hop unavailable; used counting
- Image 094: preferred between_attr unavailable; used counting
- Image 098: preferred between_attr unavailable; used counting
- Image 100: preferred attr_relation unavailable; used counting

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

| Gate | Check | Pass |
|---|---|---|
| a | every compact edge comes from the tight subgraph | 100/100 |
| b | every compact edge touches a stored core node; stored core_nodes ⊆ compact labels; v2 query's named labels + answer are compact labels | 100/100 |
| c | no duplicate triples in the compact graph | 100/100 |
| d | compact node count 4–8 (target 5–7); stored nodes list matches | 100/100 |
| e | single query's answer re-derived (counting: from the large graph), exactly one answer | 200/200 |
| f | query's key edge(s) present (or between-geometry / count holds) | 100/100 |
| g | image file is a real JPEG (> 5 KB) | 100/100 |
| h | anchors in large graph; compact graph is a subset of the large graph | 100/100 |
| i | between_attr: anchors pinnable, answer single-instance; centre-x rule recomputed | 25/25 |
| j | every compact object label is a core node or in a compact relation | 100/100 |
| k | benchmark-level: exactly 100 items, unique numbers and image ids | 1/1 |
| m | number/id/filepath agree across entry + all three JSONs; filenames match | 100/100 |
| n | no self-relation edges in the compact graph | 100/100 |

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
