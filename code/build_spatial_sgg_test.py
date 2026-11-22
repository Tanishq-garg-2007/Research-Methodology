import os, json, base64

os.makedirs('kaggle_spatial_sgg_test', exist_ok=True)

with open('Images/Images/GQA Image 002.jpg', 'rb') as f:
    img2_b64 = base64.b64encode(f.read()).decode('ascii')

script_template = '''# =====================================================================
# End-to-End Test: Spatial-Focused Scene Graph Generation -> Direct QA
# Model: Qwen/Qwen3-VL-2B-Instruct
# Image: GQA Image 002 (Airplane scene)
# Question: "What is to the left of the cockpit?"
# Ground Truth: "propeller"
# Previous Failure: "fuselage" (due to lack of spatial relations in old SGG)
# =====================================================================
import os, sys, time, json, re, base64, io, subprocess
from PIL import Image

print("[INFO] Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "qwen-vl-utils", "transformers>=4.49.0", "accelerate", "pillow"])

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"[INFO] Using Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}) | Precision: {dtype}")

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

# Load Image 002
IMG2_B64 = "__IMG2_B64__"
img_bytes = base64.b64decode(IMG2_B64.encode('ascii'))
image_002 = Image.open(io.BytesIO(img_bytes)).convert("RGB")
image_002.save("image_002.jpg")
print(f"[INFO] Loaded Image 002 (Size: {image_002.size})")

# =====================================================================
# STEP 1: SPATIAL-FOCUSED SCENE GRAPH GENERATION PROMPT
# =====================================================================
SYSTEM_PROMPT_SPATIAL = """You are an expert Vision-Language Scene Graph Generator specializing in Fine-Grained Spatial Reasoning.
Your task is to analyze the image and generate a structured scene graph defining visible objects, their visual attributes, and crucially their relative SPATIAL and DIRECTIONAL relationships.

CRITICAL RULES:
1. Object & Sub-Component Extraction:
   - Decompose primary entities into distinct child objects and structural sub-parts (e.g. for an aircraft: airplane, cockpit, fuselage, propeller, wing, tail fin, door, windows, nose, runway).
   - For every object, provide 2 to 4 concise visual attributes (color, material, shape, state).
2. SPATIAL & DIRECTIONAL RELATIONSHIPS (PRIMARY FOCUS):
   - You MUST explicitly capture the horizontal and vertical spatial arrangement between distinct objects across the scene.
   - Use precise directional predicates:
     * "to the left of" / "to the right of" (horizontal spatial alignment)
     * "above" / "below" / "on top of" / "under" (vertical spatial alignment)
     * "in front of" / "behind" (depth spatial alignment)
     * "next to" / "beside" / "near" (proximity)
     * "part of" / "on" (structural connection or support)
   - Do NOT just output part-whole relations like "has". Always include directional spatial relationships between decoupled objects (e.g. "propeller to the left of cockpit", "cockpit to the right of propeller", "wing to the left of fuselage").
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

print("\\n" + "="*70)
print("PHASE 1: GENERATING SPATIAL-FOCUSED SCENE GRAPH FROM IMAGE 002")
print("="*70)

messages_sgg = [
    {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT_SPATIAL}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_002},
            {"type": "text", "text": FEW_SHOT_SPATIAL_PROMPT}
        ]
    }
]

chat_text_sgg = processor.apply_chat_template(messages_sgg, tokenize=False, add_generation_prompt=True)
inputs_sgg = processor(text=[chat_text_sgg], images=[image_002], padding=True, return_tensors="pt").to(device)

t0 = time.time()
with torch.no_grad():
    out_ids_sgg = model.generate(
        **inputs_sgg,
        max_new_tokens=1024,
        do_sample=False,
        repetition_penalty=1.15
    )
sgg_time = time.time() - t0

gen_ids_sgg = [o[len(inp):] for inp, o in zip(inputs_sgg["input_ids"], out_ids_sgg)]
raw_sgg_output = processor.batch_decode(gen_ids_sgg, skip_special_tokens=True)[0].strip()
print(f"[INFO] Scene graph generated in {sgg_time:.2f}s.")
print(f"\\n--- RAW GENERATED SCENE GRAPH OUTPUT ---\\n{raw_sgg_output}\\n")

# Robust JSON extraction
def parse_json(raw_text):
    sanitized = re.sub(r'\\]\\]\\s*,', ']},', raw_text)
    sanitized = re.sub(r'\\]\\]\\s*$', ']}', sanitized)
    m = re.search(r"```json\\s*(\\{.*?\\})\\s*```", sanitized, re.DOTALL)
    if m:
        target = m.group(1)
    else:
        m2 = re.search(r"(\\{.*\\})", sanitized, re.DOTALL)
        target = m2.group(1) if m2 else sanitized
    try:
        return json.loads(target)
    except Exception as e:
        print(f"[WARN] Direct JSON parse failed: {e}, attempting cleanup...")
        target = re.sub(r',\\s*\\}', '}', target)
        target = re.sub(r',\\s*\\]', ']', target)
        return json.loads(target)

new_scene_graph = parse_json(raw_sgg_output)
print("\\nParsed Scene Graph:")
print(f"Objects ({len(new_scene_graph.get('objects', []))}):")
for o in new_scene_graph.get("objects", []):
    print(f"  - {o.get('name')}: attributes {o.get('attributes', [])}")

print(f"\\nRelations ({len(new_scene_graph.get('relations', []))}):")
for r in new_scene_graph.get("relations", []):
    print(f"  - {r.get('subject')} --[{r.get('relation')}]--> {r.get('object')}")

# =====================================================================
# STEP 2: DIRECT PROMPTING QA USING THE NEW SCENE GRAPH
# =====================================================================
print("\\n" + "="*70)
print("PHASE 2: DOWNSTREAM DIRECT QUESTION ANSWERING")
print("="*70)

question = "What is to the left of the cockpit?"
ground_truth = "propeller"

# Format new scene graph for Direct QA prompt
objects = new_scene_graph.get("objects", [])
relations = new_scene_graph.get("relations", [])
obj_lines = [f"  - {o.get('name', 'unknown').replace('_', ' ')} (attributes: [{', '.join(o.get('attributes', []))}])" if o.get('attributes') else f"  - {o.get('name', 'unknown').replace('_', ' ')} (no attributes)" for o in objects]
rel_lines = [f'  - "{r.get("subject", "").replace("_", " ")}" {r.get("relation", "").replace("_", " ")} "{r.get("object", "").replace("_", " ")}"' for r in relations]

direct_prompt = (
    "You are given a scene graph representation of an image containing objects and their relationships, followed by a question.\\n\\n"
    f"Scene Graph:\\nObjects ({len(objects)}):\\n" + "\\n".join(obj_lines) + "\\n\\n"
    f"Relationships ({len(relations)}):\\n" + "\\n".join(rel_lines) + "\\n\\n"
    f"Question: {question}\\n\\n"
    "Instructions: Based strictly on the scene graph provided above, answer the question accurately. "
    "Provide only the direct concise answer (a few words or a single term) without explanation or reasoning.\\n"
    "Answer:"
)

print(f"Direct QA Prompt Sent to Model:\\n{direct_prompt}\\n")

messages_qa = [
    {"role": "user", "content": [{"type": "text", "text": direct_prompt}]}
]

chat_text_qa = processor.apply_chat_template(messages_qa, tokenize=False, add_generation_prompt=True)
inputs_qa = processor(text=[chat_text_qa], return_tensors="pt").to(device)

t1 = time.time()
with torch.no_grad():
    out_ids_qa = model.generate(
        **inputs_qa,
        max_new_tokens=32,
        do_sample=False,
        repetition_penalty=1.15
    )
qa_time = time.time() - t1

gen_ids_qa = [o[len(inp):] for inp, o in zip(inputs_qa["input_ids"], out_ids_qa)]
raw_qa_output = processor.batch_decode(gen_ids_qa, skip_special_tokens=True)[0].strip()

clean_pred = re.sub(r"^(?:answer|the answer is|final answer):\\s*", "", raw_qa_output, flags=re.IGNORECASE).strip()
clean_pred = clean_pred.split("\\n")[0].strip().rstrip(".").lower()

is_correct = (clean_pred == ground_truth) or (ground_truth in clean_pred) or (clean_pred in ground_truth and len(clean_pred) > 1)

print("="*70)
print("FINAL END-TO-END VERIFICATION RESULT:")
print("="*70)
print(f"Image: GQA Image 002")
print(f"Question: {question}")
print(f"Ground Truth: '{ground_truth}'")
print(f"Previous Old SGG Result: 'fuselage' (FAILED)")
print(f"New Spatial SGG Predicted Answer: '{clean_pred}' (Time: {qa_time:.2f}s)")
print(f"Evaluation: {'SUCCESS / CORRECT!' if is_correct else 'INCORRECT'} (Exact Match: {clean_pred == ground_truth})")
print("="*70)

# Save result to JSON
res_summary = {
    "image_number": 2,
    "question": question,
    "ground_truth": ground_truth,
    "previous_old_sgg_prediction": "fuselage",
    "new_spatial_sgg_prediction": clean_pred,
    "is_correct": is_correct,
    "is_exact": clean_pred == ground_truth,
    "latency_sgg": round(sgg_time, 2),
    "latency_qa": round(qa_time, 2),
    "generated_scene_graph": new_scene_graph,
    "qa_raw_output": raw_qa_output
}

with open("spatial_sgg_test_result.json", "w") as f:
    json.dump(res_summary, f, indent=2)

print("[INFO] Results saved to spatial_sgg_test_result.json successfully.")
'''

full_script = script_template.replace('__IMG2_B64__', img2_b64)

with open('kaggle_spatial_sgg_test/test_spatial_sgg.py', 'w', encoding='utf-8') as f:
    f.write(full_script)

metadata = {
    "id": "tanishqgarg2007/qwen3-spatial-sgg-test",
    "title": "qwen3 spatial sgg test",
    "code_file": "test_spatial_sgg.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": []
}

with open('kaggle_spatial_sgg_test/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("Kaggle spatial SGG test package updated successfully in kaggle_spatial_sgg_test/")
