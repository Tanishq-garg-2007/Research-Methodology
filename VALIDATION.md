# VALIDATION.md — GQA-100 spatial benchmark (Exp-07 dataset)

Built 2026-11-21. Source: Hugging Face `Voxel51/GQA-Scene-Graph` (GQA 35k subset).
Layout mirrors the Drive "RM" folder: `Images/`, `Large Scene Graph/`,
`Small Scene Graph/`, `Queries/` (files `GQA Image 001` … `GQA Image 100`),
plus `benchmark.json` at the root. Nothing here has been uploaded to Drive.

## What was built

- **100 images, 100 questions** — exactly one query per image.
- Per image: the full original graph (Large), the compact local graph (Small),
  the single query with answer + derivation (Queries), and the image itself.
- Anchor selection is automatic (labels ranked by relation involvement; first
  pair yielding a qualifying compact query wins).
- Template mix: `on` 23, `left_of` 57, `right_of` 13, `between` 7, `with` 0.
  (`with` is last in the try order, so it almost never wins.)
- Compact node counts: 6 nodes × 78 items, 5 × 18, 4 × 4 → **96/100 inside the
  5–7 target**; the 4 items at 4 nodes used the documented fallback (no
  distractor could be admitted without breaking answer uniqueness).
- Compact relation counts: min 4, median 13, max 31. 37 items in the 8–12
  target band, **53 above 12**, 10 below 8. The compact rule bounds nodes,
  not edges; hub labels (e.g. roof/building in image 084, 28 relations) make
  dense small graphs. See "Known limitations".
- Full graphs: 6–41 objects, 4–113 relations per image.

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
| b | every compact edge touches a core node; stored core_nodes correct | 100/100 |
| c | no duplicate triples in the compact graph | 100/100 |
| d | compact node count 4–8 (target 5–7); stored nodes list matches | 100/100 |
| e | single query's answer re-derived from the compact graph, exactly one answer | 200/200 |
| f | query's key edge present (or between-geometry holds) | 100/100 |
| g | image file is a real JPEG (> 5 KB) | 100/100 |
| h | anchors in large graph; compact graph is a subset of the large graph | 100/100 |
| i | between-queries: anchors + answer single-instance; centre-x rule recomputed | 7/7 |
| j | every compact object label is a core node or in a compact relation | 100/100 |
| k | benchmark-level: exactly 100 items, unique numbers and image ids | 1/1 PASS |
| m | number/id/filepath agree across entry + all three JSONs; filenames match | 100/100 |
| n | no self-relation edges in the compact graph | 100/100 |

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
tomato + fork, the compact graph is **6 nodes / 11 relations** and the query
is "What is on the plate?" → fork. (This build dedupes relations, so its
tight counts read lower than the walkthrough's 35-with-duplicates; the
compact 6/11 reproduce exactly.) Item 001 in the shipped benchmark uses the
automatic anchor pair plate + glass, the same query, 6 nodes, and 7 relations
after the self-edge exclusion.

## Hand-verification (5 random items, seed 7)

Each item's query was checked against its image and its graph (boxes inspected
where the eye alone could not settle it). All 5 PASS:

- **042 (gqa 2323139)** — "What is on the bed?" → pillow. PASS. Two dogs on a
  bed; pillows are clearly on the bed.
- **020 (gqa 4114)** — "What is between the plant and the chair?" → window.
  PASS. Boxes confirm: plant centre-x 0.37 < window 0.525 < chair 0.615; the
  plant and chair are visible through the bay window's left/right panes.
- **051 (gqa 2340367)** — "What is to the left of the glass?" → man. PASS
  after box check: the annotated glass is the tiny one at the far-right edge
  [0.97, 0.56]; the men stand left of it. Caveat: the "man" box is over-broad
  (covers 91% of the width — a group box), though the relation direction is
  correct.
- **084 (gqa 3773)** — "What is to the right of the box?" → roof. PASS. The
  red-tiled roofs extend right of the small chimney-boxes on the lower roof.
  This is the densest compact graph (28 relations).
- **007 (gqa 2375631)** — "What is to the right of the lid?" → counter.
  PASS. Toilet lid bottom-left; sink counter clearly to its right.

Earlier round (seed 42, on the superseded tight-subgraph build) found the
noise that motivated filters F1/F3 and the self-edge exclusion: self-relations
("trees left of trees"), same subject+object annotated both directions, and
speck-sized boxes. Those patterns are now filtered at generation time and
gated at validation time (gates b/e/n).

## Known limitations (disclosed, not hidden)

- **Relation-count tail.** 53/100 compact graphs have more than 12 relations
  (max 31). The rule bounds nodes, not edges; hub labels produce dense small
  graphs. If a ~12 cap is wanted, it needs a new rule (e.g. keep the key edge
  plus a per-node cap) — not added unilaterally.
- **Label ambiguity (the known GQA weakness).** Relations point to labels,
  not unique instances. 51/100 queries are fully unambiguous (every involved
  label has exactly one instance in the image); **49/100 involve a label with
  multiple instances**. Every answer is still uniquely derivable from its
  compact graph (gate e proves it), but for those 49 the question is
  answerable from the graph, not necessarily from the image alone.
- **Template imbalance.** No `with` queries (last in try order); `left_of`
  dominates (57/100) because GQA relations are overwhelmingly left/right.
- **GQA annotation noise.** Boxes can be over-broad (image 051's "man") and
  some relations are box artefacts. The validator proves graph-faithfulness,
  not visual truth; the 5-item hand check is the visual sample.

## Files

- `build_gqa_bench100.py` — the build script (streaming, compact rule, layout)
- `validate_bench100.py` — the independent validator
- `benchmark.json` — one entry per image (number, ids, anchors, query, counts, paths)
- `build_log.json` — scan/skip/download counts
- `validation_report.json` — machine-readable gate results
- `VALIDATION.md` — this file
- `Images/GQA Image 001.jpg` … `GQA Image 100.jpg`
- `Large Scene Graph/GQA Image 001 - Large Scene Graph.json` … (full graph)
- `Small Scene Graph/GQA Image 001 - Small Scene Graph.json` … (compact graph)
- `Queries/GQA Image 001 - Queries.json` … (exactly one query object each)
