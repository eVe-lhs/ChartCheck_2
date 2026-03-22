import json
import os
import torch
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILES = [
    "test_1_wo_spec.json", 
    "test_2_wo_spec.json", 
    "train_wo_spec.json", 
    "validation_wo_spec.json"
]
CACHE_FILE = "grounding_cache.json"

# --- STORAGE & RESUME LOGIC ---
def load_data(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"Warning: {filepath} not found. Skipping.")
    return []

def load_cache(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4)

if __name__ == "__main__":
    print("--- Starting Qwen2.5-VL Visual Grounding ---")
    
    # 1. Aggregate unique images
    all_unique_images = set()
    for filepath in INPUT_FILES:
        dataset = load_data(filepath)
        for item in dataset:
            if "local_image_path" in item:
                all_unique_images.add(item["local_image_path"])
    
    unique_images = list(all_unique_images)
    print(f"Total unique images found across dataset: {len(unique_images)}")

    # 2. Load Cache
    cache = load_cache(CACHE_FILE)
    for img in unique_images:
        if img not in cache:
            cache[img] = {"deplot_table": None, "vega_lite_spec": None}
            
    # 3. Filter pending work
    pending_images = [img for img in unique_images if cache[img].get("vega_lite_spec") is None]
    
    if not pending_images:
        print("Qwen pass already complete for all images. Exiting.")
        exit()

    print(f"Loading Qwen2.5-VL-3B-Instruct... {len(pending_images)} images pending.")
    
    # 4. Load Model (bfloat16 for 12GB VRAM limits)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    prompt = """Analyze this chart and output a STRICTLY MINIMIZED Vega-Lite specification in JSON format. 
You must ONLY extract:
1. The chart type (e.g., bar, line, pie) under 'mark'.
2. The axis labels and their types (e.g., quantitative, nominal) under 'encoding'.
3. A brief trend description under a 'description' field.
4. Color information on certain series if available (e.g., add a 'color' encoding block that maps a nominal category to the specific colors used in the chart).

Do NOT extract the underlying data points. Output ONLY valid JSON."""

    # 5. Inference Loop
    for i, img_path in enumerate(tqdm(pending_images, desc="Running Qwen2.5-VL")):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            
            generated_ids = model.generate(**inputs, max_new_tokens=300)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs.input_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            cache[img_path]["vega_lite_spec"] = output_text
            
            # Save incrementally every 10 images
            if i % 10 == 0:
                save_cache(cache, CACHE_FILE)
                
        except Exception as e:
            print(f"\nError processing {img_path} with Qwen: {e}")

    # Final save
    save_cache(cache, CACHE_FILE)
    print("Qwen2.5-VL extraction complete!")