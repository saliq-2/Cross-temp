from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("uclanlp/TemMed-Bench", "Image Pair Selection")
# print(ds)


# Login using e.g. `huggingface-cli login` to access this dataset
ds_Knowledge_corpus = load_dataset("uclanlp/TemMed-Bench", "TrainSet KnowledgeCorpus")
# print(ds_Knowledge_corpus)
#vqa and report generation


# Login using e.g. `huggingface-cli login` to access this dataset
ds_vqa = load_dataset("uclanlp/TemMed-Bench", "VQA & Report Generation")
# print(ds_vqa)


import os
import torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score







# ==========================================
# 1. CONFIGURATION
# ==========================================
# Set DEMO_MODE = True if you do not have the 500GB CheXpert images downloaded locally.
# The code will simulate the vision part so you can test the Logic Model immediately.
DEMO_MODE = True  

# If you HAVE the images, set this to your folder path (e.g. "/content/CheXpert-v1.0")
BASE_IMAGE_DIR = "./" 

# ==========================================
# 2. LOAD MODELS (4-bit for efficiency)
# ==========================================
print("Loading Models...")
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

# Stage 1: The Eyes (LLaVA-v1.6)
vision_processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
vision_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", 
    quantization_config=quant_config, 
    device_map="auto"
)

# Stage 2: The Brain (Phi-3-Mini)
text_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
text_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    quantization_config=quant_config, 
    device_map="auto"
)
# Fix padding token issue
if text_tokenizer.pad_token is None: text_tokenizer.pad_token = text_tokenizer.eos_token
text_model.config.pad_token_id = text_tokenizer.eos_token_id

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def generate_medical_caption(image, model, processor):
    """Generates a targeted medical description using keywords from the paper."""
    # We ask specifically for pathologies mentioned in TEMMED Appendix A.3
    prompt = "[INST] <image>\nAnalyze this chest X-ray for: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion. Describe findings concisely. [/INST]"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=120)
    return processor.decode(out[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

def reason_about_change(old_cap, new_cap, question, model, tokenizer):
    """Compares two captions to answer the temporal question."""
    msg = [{"role": "user", "content": f"Compare these reports:\nOLD: {old_cap}\nNEW: {new_cap}\nQUESTION: {question}\nAnswer 'Yes' or 'No'."}]
    inp = tokenizer.apply_chat_template(msg, return_tensors="pt", add_generation_prompt=True, return_dict=True).to("cuda")
    out = model.generate(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'], max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inp['input_ids'].shape[-1]:], skip_special_tokens=True)

def load_local_image(path_str):
    try:
        return Image.open(os.path.join(BASE_IMAGE_DIR, path_str)).convert("RGB")
    except:
        return None

# ==========================================
# 4. LOAD & EVALUATE
# ==========================================
print("Loading VQA Dataset...")
# This matches the LAST load in your logs (the correct one)
ds = load_dataset("uclanlp/TemMed-Bench", "VQA & Report Generation", split="test")

preds, truths = [], []

print("Starting Evaluation Pipeline...")
# We test on the first 10 items for now
for i, item in enumerate(tqdm(ds.select(range(10)))):
    try:
        # 1. GET DATA
        # These keys match your logs exactly:
        hist_path = item['historical_image_path']
        curr_path = item['image_path']
        
        # 'VQA_Final' contains the list of questions for this image pair
        vqa_list = item['VQA_Final'] 
        # Handle case where it might be a single dict instead of a list
        if isinstance(vqa_list, dict): vqa_list = [vqa_list]

        # 2. RUN VISION STAGE (Once per image pair)
        if DEMO_MODE:
            # Simulated captions to prove Stage 2 works
            cap_hist = "The lungs show mild interstitial edema and cardiomegaly."
            cap_curr = "The edema has resolved. Heart size is stable."
        else:
            img_h = load_local_image(hist_path)
            img_c = load_local_image(curr_path)
            
            if img_h is None or img_c is None:
                # Skip if you don't have the images downloaded
                continue
                
            cap_hist = generate_medical_caption(img_h, vision_model, vision_processor)
            cap_curr = generate_medical_caption(img_c, vision_model, vision_processor)

        # 3. RUN LOGIC STAGE (For every question about this pair)
        for qa in vqa_list:
            question = qa['question']
            truth = qa['answer']
            
            pred = reason_about_change(cap_hist, cap_curr, question, text_model, text_tokenizer)
            
            # Normalize to simple yes/no
            p_clean = "yes" if "yes" in pred.lower() else "no"
            t_clean = "yes" if "yes" in truth.lower() else "no"
            
            preds.append(p_clean)
            truths.append(t_clean)

            # Print first sample to verify
            if len(preds) == 1:
                print(f"\n[SAMPLE] Q: {question}")
                print(f"Logic: {p_clean} | Truth: {t_clean}")

    except Exception as e:
        print(f"Error on sample {i}: {e}")

# ==========================================
# 5. PRINT METRICS
# ==========================================
if len(preds) > 0:
    acc = accuracy_score(truths, preds) * 100
    f1 = f1_score(truths, preds, pos_label='yes') * 100
    print(f"\nResults on {len(preds)} questions:")
    print(f"Accuracy: {acc:.2f}% (Baseline to beat: 51.65%)")
    print(f"F1 Score: {f1:.2f}%")
else:
    print("\nNo predictions made. If you don't have images, ensure DEMO_MODE = True")