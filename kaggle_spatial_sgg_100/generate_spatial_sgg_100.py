# =====================================================================
# Full 100-Image Spatial-Focused Scene Graph Generation Benchmark
# Model: Qwen/Qwen3-VL-2B-Instruct
# Strategy: Few-Shot + Embedded Spatial CoT
# Output: scene_graphs_100_images_spatial.json
# =====================================================================
import os, sys, time, json, re, zipfile, subprocess
from PIL import Image

print("[INFO] Installing required dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "qwen-vl-utils", "transformers>=4.49.0", "accelerate", "pillow"])

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"[INFO] Using Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}) | Precision: {dtype}")

# 1. Locate and extract 100 benchmark images
IMG_DIR = "/kaggle/working/images"
os.makedirs(IMG_DIR, exist_ok=True)

print("[INFO] Scanning filesystem for images and zip archives...")

# Search locations: local directory and all of /kaggle/input
search_dirs = [".", "/kaggle/src", "/kaggle/working"]
if os.path.exists("/kaggle/input"):
    print("[INFO] Datasets in /kaggle/input:", os.listdir("/kaggle/input"))
    search_dirs.append("/kaggle/input")

for base in search_dirs:
    if not os.path.exists(base):
        continue
    for root, dirs, files in os.walk(base):
        for f in files:
            f_lower = f.lower()
            if f_lower == "images_100.zip" or (f_lower.endswith(".zip") and "image" in f_lower):
                zip_path = os.path.join(root, f)
                print(f"[INFO] Found zip archive: {zip_path}. Extracting to {IMG_DIR}...")
                try:
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(IMG_DIR)
                except Exception as e:
                    print(f"[WARN] Error extracting {zip_path}: {e}")
            elif f_lower.endswith((".jpg", ".jpeg", ".png")):
                src = os.path.join(root, f)
                dst = os.path.join(IMG_DIR, f)
                if not os.path.exists(dst):
                    try:
                        os.link(src, dst)
                    except Exception:
                        import shutil
                        shutil.copy2(src, dst)

available_images = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
print(f"[INFO] Successfully found {len(available_images)} images in {IMG_DIR}.")
if len(available_images) > 0:
    print(f"[INFO] Sample images: {available_images[:3]} ... {available_images[-2:]}")
else:
    raise RuntimeError(f"No images found in {IMG_DIR}. Check directory scans.")

# 2. Load Model
model_id = "Qwen/Qwen3-VL-2B-Instruct"
print(f"[INFO] Loading model {model_id}...")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("[INFO] Model loaded successfully.")

# 3. Spatial Prompts
SYSTEM_PROMPT_SPATIAL = """You are an expert Vision-Language Scene Graph Generator specializing in Fine-Grained Spatial Reasoning.
Your task is to analyze the image and generate a structured scene graph defining visible objects, their visual attributes, and crucially their relative SPATIAL and DIRECTIONAL relationships.

CRITICAL RULES:
1. Object & Sub-Component Extraction:
   - Decompose primary entities into distinct child objects and structural sub-parts (e.g. for vehicles: cab, cargo, wheels, headlights; for aircraft: fuselage, cockpit, propeller, wing, tail fin; for rooms/streets: distinct furniture, fixtures, road elements).
   - For every object, provide 2 to 4 concise visual attributes (color, material, shape, state).
2. SPATIAL & DIRECTIONAL RELATIONSHIPS (PRIMARY FOCUS):
   - You MUST explicitly capture the horizontal and vertical spatial arrangement between distinct objects across the scene.
   - Use precise directional predicates:
     * "to the left of" / "to the right of" (horizontal spatial alignment)
     * "above" / "below" / "on top of" / "under" (vertical spatial alignment)
     * "in front of" / "behind" (depth spatial alignment)
     * "next to" / "beside" / "near" (proximity)
     * "part of" / "on" (structural connection or support)
   - Do NOT just output part-whole relations like "has". Always include directional spatial relationships between decoupled objects (e.g. "object_A to the left of object_B", "object_B to the right of object_A").
3. Output Format: Output ONLY valid JSON matching the schema below. No conversational text."""

FEW_SHOT_SPATIAL_PROMPT = """Analyze this image and generate the complete spatial scene graph in valid JSON.

Target Schema:
```json
{
  "reasoning_steps": [
    "1. Scan the global scene and identify key objects and their sub-components...",
    "2. Isolate distinct visual attributes for each object...",
    "3. Determine horizontal, vertical, and depth spatial relations (to the left of, to the right of, above, below)..."
  ],
  "objects": [
    {"name": "object_name", "attributes": ["attr1", "attr2"]}
  ],
  "relations": [
    {"subject": "object_A", "relation": "to the left of", "object": "object_B"},
    {"subject": "object_B", "relation": "to the right of", "object": "object_A"}
  ]
}
```

Example 1 (Outdoor Vehicle Scene):
Output:
```json
{
  "reasoning_steps": [
    "1. The scene features a delivery truck parked near a warehouse loading bay.",
    "2. Decomposing components: driver cab, cargo container, headlights, wheels, and loading platform.",
    "3. Spatial relations: driver cab is in front of cargo container, wheels are below cab, loading platform is to the right of the truck."
  ],
  "objects": [
    {"name": "truck", "attributes": ["large", "white and blue", "commercial"]},
    {"name": "driver cab", "attributes": ["white", "glass windows", "front section"]},
    {"name": "cargo container", "attributes": ["blue", "rectangular", "metal"]},
    {"name": "headlight", "attributes": ["bright", "clear glass", "circular"]},
    {"name": "wheel", "attributes": ["black", "rubber", "circular"]},
    {"name": "loading platform", "attributes": ["concrete", "gray", "elevated"]}
  ],
  "relations": [
    {"subject": "driver cab", "relation": "in front of", "object": "cargo container"},
    {"subject": "cargo container", "relation": "behind", "object": "driver cab"},
    {"subject": "headlight", "relation": "part of", "object": "driver cab"},
    {"subject": "wheel", "relation": "below", "object": "driver cab"},
    {"subject": "truck", "relation": "to the left of", "object": "loading platform"},
    {"subject": "loading platform", "relation": "to the right of", "object": "truck"}
  ]
}
```

Example 2 (Aviation / Runway Scene):
Output:
```json
{
  "reasoning_steps": [
    "1. The image shows a twin-engine aircraft on a tarmac runway.",
    "2. Decomposing parts: fuselage, cockpit, wing, left propeller, right propeller, tail fin, and tarmac.",
    "3. Spatial arrangement: cockpit is in the forward upper fuselage, left propeller is to the left of the cockpit, right propeller is to the right of the cockpit, wing extends horizontally, tarmac is below."
  ],
  "objects": [
    {"name": "airplane", "attributes": ["passenger", "twin-engine", "white"]},
    {"name": "fuselage", "attributes": ["cylindrical", "white body", "long"]},
    {"name": "cockpit", "attributes": ["front cabin", "tinted glass", "upper"]},
    {"name": "left propeller", "attributes": ["spinning blades", "mounted on wing", "dark"]},
    {"name": "right propeller", "attributes": ["spinning blades", "mounted on wing", "dark"]},
    {"name": "tail fin", "attributes": ["vertical", "blue logo", "rear"]},
    {"name": "tarmac", "attributes": ["flat", "gray asphalt", "ground"]}
  ],
  "relations": [
    {"subject": "cockpit", "relation": "part of", "object": "fuselage"},
    {"subject": "left propeller", "relation": "to the left of", "object": "cockpit"},
    {"subject": "cockpit", "relation": "to the right of", "object": "left propeller"},
    {"subject": "right propeller", "relation": "to the right of", "object": "cockpit"},
    {"subject": "cockpit", "relation": "to the left of", "object": "right propeller"},
    {"subject": "tail fin", "relation": "behind", "object": "cockpit"},
    {"subject": "airplane", "relation": "on top of", "object": "tarmac"},
    {"subject": "tarmac", "relation": "below", "object": "airplane"}
  ]
}
```

Now analyze this image and output the complete scene graph with explicit spatial relationships in valid JSON:"""

def parse_json(raw_text):
    sanitized = re.sub(r'\]\]\s*,', ']},', raw_text)
    sanitized = re.sub(r'\]\]\s*$', ']}', sanitized)
    m = re.search(r"```json\s*(\{.*?\})\s*```", sanitized, re.DOTALL)
    if m:
        target = m.group(1)
    else:
        m2 = re.search(r"(\{.*\})", sanitized, re.DOTALL)
        target = m2.group(1) if m2 else sanitized
    try:
        return json.loads(target), True
    except Exception:
        try:
            target = re.sub(r',\s*\}', '}', target)
            target = re.sub(r',\s*\]', ']', target)
            return json.loads(target), True
        except Exception as e:
            return {"raw": raw_text, "error": str(e)}, False

# 4. Checkpointed Generation Loop for 100 Images
OUT_FILE = "scene_graphs_100_images_spatial.json"
results = {}

if os.path.exists(OUT_FILE):
    try:
        with open(OUT_FILE, "r") as f:
            for item in json.load(f):
                results[item["image_number"]] = item
        print(f"[RESUME] Loaded {len(results)} previously generated scene graphs.")
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint: {e}")

print("\n" + "="*70)
print(f"STARTING 100-IMAGE SPATIAL SCENE GRAPH GENERATION (Remaining: {100 - len(results)})")
print("="*70)

t_start_total = time.time()

for img_idx in range(1, 101):
    fname = f"GQA Image {img_idx:03d}.jpg"
    img_path = os.path.join(IMG_DIR, fname)

    if img_idx in results:
        continue

    if not os.path.exists(img_path):
        print(f"[{img_idx:03d}/100] [WARN] File not found: {fname}")
        continue

    t0 = time.time()
    try:
        pil_img = Image.open(img_path).convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_SPATIAL}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": FEW_SHOT_SPATIAL_PROMPT}
                ]
            }
        ]

        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat_text], images=[pil_img], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                repetition_penalty=1.15
            )
        elapsed = time.time() - t0

        trimmed_ids = out_ids[0][len(inputs.input_ids[0]):]
        out_text = processor.decode(trimmed_ids, skip_special_tokens=True).strip()
        parsed, is_valid = parse_json(out_text)

        obj_count = len(parsed.get("objects", [])) if is_valid else 0
        rel_count = len(parsed.get("relations", [])) if is_valid else 0

        directional_preds = {"to the left of", "to the right of", "above", "below", "in front of", "behind", "next to", "beside", "near"}
        dir_count = sum(1 for r in parsed.get("relations", []) if r.get("relation", "").lower().strip() in directional_preds) if is_valid else 0

        entry = {
            "image_number": img_idx,
            "filename": fname,
            "latency_sec": round(elapsed, 2),
            "tokens": len(trimmed_ids),
            "is_valid_json": is_valid,
            "objects_count": obj_count,
            "relations_count": rel_count,
            "directional_relations_count": dir_count,
            "scene_graph": parsed
        }

        results[img_idx] = entry
        print(f"[{img_idx:03d}/100] {fname} | {elapsed:.1f}s | {len(trimmed_ids)} tok | Valid: {is_valid} | Objs: {obj_count} | Rels: {rel_count} (Spatial: {dir_count})")

    except Exception as exc:
        print(f"[{img_idx:03d}/100] [ERROR] {fname}: {exc}")

    if img_idx % 5 == 0 or img_idx == 100:
        sorted_results = [results[k] for k in sorted(results.keys())]
        try:
            with open(OUT_FILE, "w", encoding="utf-8") as f:
                json.dump(sorted_results, f, indent=2)
        except Exception as e:
            print(f"[WARN] Checkpoint write failed: {e}")

total_time = time.time() - t_start_total
sorted_final = [results[k] for k in sorted(results.keys())]
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(sorted_final, f, indent=2)

print("\n" + "="*70)
print(f"COMPLETED ALL {len(sorted_final)}/100 SPATIAL SCENE GRAPHS")
print(f"Total Time: {total_time:.1f}s ({total_time/60:.2f} min) | Avg per image: {total_time/max(len(sorted_final), 1):.2f}s")
print(f"Saved to: {OUT_FILE}")
print("="*70)
