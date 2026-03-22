import json
import os
import torch
import gc
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILES = [
    "test1_wo_spec.json", 
    "test2_wo_spec.json", 
    "train_wo_spec.json", 
    "validation_wo_spec.json"
]

DEPLOT_CACHE_FILE = "deplot_cache.json"
QWEN_CACHE_FILE = "qwen_cache.json"

# --- 1. STORAGE & RESUME LOGIC ---
def load_data(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
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

# Aggregate all unique images across all dataset files
all_unique_images = set()
total_claims = 0

for filepath in INPUT_FILES:
    dataset = load_data(filepath)
    total_claims += len(dataset)
    for item in dataset:
        if "local_image_path" in item:
            all_unique_images.add(item["local_image_path"])

unique_images = list(all_unique_images)
print(f"Total claims across all files: {total_claims}")
print(f"Total unique images to process: {len(unique_images)}")

# Load both caches independently
deplot_cache = load_cache(DEPLOT_CACHE_FILE)
qwen_cache = load_cache(QWEN_CACHE_FILE)


# --- 2. DEPLOT EXTRACTION (Numeric Grounding) ---
def run_deplot_pass(images, cache, cache_filename):
    # Check if there's remaining work before loading the model
    pending_images = [img for img in images if img not in cache]
    if not pending_images:
        print("DePlot pass already complete for all images.")
        return

    print(f"Loading DePlot... {len(pending_images)} images pending.")
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
    
    processor = Pix2StructProcessor.from_pretrained("google/deplot")
    model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot").to("cuda")
    
    for img_path in tqdm(pending_images, desc="Running DePlot"):
        try:
            image = Image.open(img_path)
            inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt").to("cuda")
            predictions = model.generate(**inputs, max_new_tokens=512)
            table = processor.decode(predictions[0], skip_special_tokens=True)
            
            cache[img_path] = table
            
            # Save incrementally (resume logic) - saves every 10 images
            if pending_images.index(img_path) % 10 == 0:
                save_cache(cache, cache_filename)
            
        except Exception as e:
            print(f"Error processing {img_path} with DePlot: {e}")
            
    # Final save for the pass
    save_cache(cache, cache_filename)
    
    # Completely flush VRAM before loading the next model
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    print("DePlot pass complete. VRAM flushed.")


# --- 3. VLM EXTRACTION (Visual Grounding & Color Info) ---
def run_qwen_pass(images, cache, cache_filename):
    # Check if there's remaining work before loading the model
    pending_images = [img for img in images if img not in cache]
    if not pending_images:
        print("Qwen2.5-VL pass already complete for all images.")
        return

    print(f"Loading Qwen2.5-VL... {len(pending_images)} images pending.")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    # Load 3B model in bfloat16 to fit comfortably in 12GB VRAM
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    # Enforcing strict extraction and color information
    prompt = """Analyze this chart and output a STRICTLY MINIMIZED Vega-Lite specification in JSON format. 
You must ONLY extract:
1. The chart type (e.g., bar, line, pie) under 'mark'.
2. The axis labels and their types (e.g., quantitative, nominal) under 'encoding'.
3. A brief trend description under a 'description' field.
4. Color information on certain series if available (e.g., add a 'color' encoding block that maps a nominal category to the specific colors used in the chart).

Do NOT extract the underlying data points. Output ONLY valid JSON."""

    for img_path in tqdm(pending_images, desc="Running Qwen2.5-VL"):
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
            
            # Generate the spec
            generated_ids = model.generate(**inputs, max_new_tokens=300)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs.input_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            cache[img_path] = output_text
            
            # Save incrementally - saves every 10 images
            if pending_images.index(img_path) % 10 == 0:
                save_cache(cache, cache_filename)
            
        except Exception as e:
            print(f"Error processing {img_path} with Qwen: {e}")

    # Final save for the pass
    save_cache(cache, cache_filename)

    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    print("Qwen2.5-VL pass complete. VRAM flushed.")


# --- 4. EXECUTION ---
if __name__ == "__main__":
    print("Starting Phase 1 Dual-File Grounding Pipeline...")
    
    # Pass 1: Numeric Grounding
    run_deplot_pass(unique_images, deplot_cache, DEPLOT_CACHE_FILE)
    
    # Pass 2: Visual Grounding
    run_qwen_pass(unique_images, qwen_cache, QWEN_CACHE_FILE)
    
    print("All extraction complete! Data is stored in 'deplot_cache.json' and 'qwen_cache.json'.")