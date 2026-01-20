import sys
import os
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer, AutoModel
import numpy as np
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import json
import random

# 2x A100 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

random.seed(42)

# Set model paths
base_model_name = "Qwen/Qwen3-Embedding-8B" 
lora_checkpoint = "./lora_checkpoints/qwen_lora_epoch_9"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(">>> Loading base model and automatically dispatching to 2x A100...")

max_memory_map = {0: "78GiB", 1: "78GiB"}

model = AutoModel.from_pretrained(
    base_model_name, 
    device_map="auto", 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, # Use bfloat16 for A100s
    max_memory=max_memory_map   # Mapped for 2 GPUs
)

print(">>> Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_checkpoint)
model.eval()

# Embedding function
def get_embeddings(texts, model, tokenizer, batch_size=32):
    all_embeddings = []
    
    if hasattr(model, "device"):
         target_device = model.device
    else:
         target_device = list(model.parameters())[0].device

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # [B, L, H]
            
            embeddings = last_hidden[:, -1, :] 

            embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Move to CPU and float32 for storage/numpy conversion
        all_embeddings.append(embeddings.cpu().to(torch.float32)) 

    return torch.cat(all_embeddings, dim=0).numpy()

#Load Dataset
base_path = "./CSTS-data"
dataset = load_dataset(
    'csv', 
    data_files={
        'train': base_path + '/csts_train_reannotated.csv',
        'validation': base_path + '/csts_validation_reannotated.csv',
    }
)

# Split Validation and Test
validation = dataset['validation']
dataset['validation'] = validation.select(range(1983))
dataset['test'] = validation.select(range(1983, len(validation)))

# Construct Embedding Matrix
def build_matrix(dataset, model, tokenizer, batch_size=64):
    sent2row = {}
    texts_to_encode = []   
    rowid = 0
    dropped_instances = 0
    conditions_set = set()

    print(">>> Preparing texts for encoding...")
    for inst in tqdm(dataset):
        if inst['label'] == -1: 
            dropped_instances += 1
            continue

        c = inst['condition']
        conditions_set.add(c)

        # 1. Handle Condition
        if c not in sent2row:
            sent2row[c] = rowid
            rowid += 1
            texts_to_encode.append(c)

        # 2. Handle Sentence + Condition Context
        for s in [inst['sentence1'], inst['sentence2']]:
            unique_key = f"{c} {s}"
            
            if unique_key not in sent2row:
                sent2row[unique_key] = rowid
                rowid += 1
                
                # CSTS Prompt
                prompt = f"Retrieve semantically similar text to a given {c}, under the given: {s}"
                texts_to_encode.append(prompt)

    print(f"Dropped instances: {dropped_instances}")
    print(f"Total texts to encode: {len(texts_to_encode)}")
    
    row2sent = {rid: txt for txt, rid in sent2row.items()}

    embed_M = get_embeddings(texts_to_encode, model, tokenizer, batch_size=batch_size)

    return embed_M, sent2row, row2sent

combined_dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

M, sent2row, row2sent = build_matrix(combined_dataset, model, tokenizer, batch_size=128) 

# Save results
output_folder = "qwen-8B-finetuned-embedding"
base_save_path = './work'
save_path = os.path.join(base_save_path, output_folder)
os.makedirs(save_path, exist_ok=True)

print(f">>> Saving results to {save_path}")
np.save(os.path.join(save_path, 'M.npy'), M)
with open(os.path.join(save_path, 'sent2row.json'), "w") as f:
    json.dump(sent2row, f, indent=4)
with open(os.path.join(save_path, 'row2sent.json'), "w") as f:
    json.dump(row2sent, f, indent=4)

print("Done!")