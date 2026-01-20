import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from scipy.stats import spearmanr
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import os
from accelerate import Accelerator

# Define Checkpoint Directory
CHECKPOINT_DIR = "./lora_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")

# LoRA Configuration
LORA_R = 32         
LORA_ALPHA = 32      
LORA_DROPOUT = 0.05  
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, 
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=TARGET_MODULES,
)

# Accelerator Initialization
accelerator = Accelerator()
device = accelerator.device 
num_gpus = accelerator.num_processes 
if accelerator.is_main_process:
    print(f"Accelerator initialized. Running on {num_gpus} processes.")
    if device.type == 'cuda':
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Set environment variables for optimal multi-GPU performance
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load dataset
base_path = "./CSTS-data"
dataset_dict = load_dataset(
  'csv', 
  data_files=
  {
    'train': base_path + '/csts_train_reannotated.csv',
    'validation':  base_path + '/csts_validation_reannotated.csv',
  },
  split={
        'train': 'train',
        'validation': 'validation[:1983]', 
        'test': 'validation[1983:]',
    }
)

print(dataset_dict)

class CSTSDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, maxlen=256):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        condition = item["condition"]
        s1 = f"Retrieve semantically similar text to a given {condition}, under the given: {item['sentence1']}"
        s2 = f"Retrieve semantically similar text to a given {condition}, under the given: {item['sentence2']}"
        label = float(item["label"]) / 5.0  # normalize to [0,1]

        s1_enc = self.tokenizer(s1, truncation=True, max_length=self.maxlen,
                                padding="max_length", return_tensors=None)
        s2_enc = self.tokenizer(s2, truncation=True, max_length=self.maxlen,
                                padding="max_length", return_tensors=None)

        return {
            "s1_input_ids": s1_enc["input_ids"],
            "s1_attention_mask": s1_enc["attention_mask"],
            "s2_input_ids": s2_enc["input_ids"],
            "s2_attention_mask": s2_enc["attention_mask"],
            "label": label
        }


# Load tokenizer
model_name = "Qwen/Qwen3-Embedding-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

# Create datasets
train_dataset = CSTSDataset(dataset_dict["train"], tokenizer)
val_dataset   = CSTSDataset(dataset_dict["validation"], tokenizer)
test_dataset  = CSTSDataset(dataset_dict["test"], tokenizer)

# Adjust batch sizes for multi-GPU training
base_batch_size = 8
val_batch_size = 16

def collate_fn(batch):
    """Custom collate function to properly batch the data"""
    s1_input_ids = torch.tensor([item["s1_input_ids"] for item in batch])
    s1_attention_mask = torch.tensor([item["s1_attention_mask"] for item in batch])
    s2_input_ids = torch.tensor([item["s2_input_ids"] for item in batch])
    s2_attention_mask = torch.tensor([item["s2_attention_mask"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)
    
    return {
        "s1_input_ids": s1_input_ids,
        "s1_attention_mask": s1_attention_mask,
        "s2_input_ids": s2_input_ids,
        "s2_attention_mask": s2_attention_mask,
        "label": labels
    }

# shuffle=True is important for DistributedSampler
train_loader = DataLoader(train_dataset, batch_size=base_batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=4, pin_memory=True, collate_fn=collate_fn)

class SimilarityModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def encode(self, batch_dict):
        outputs = self.base(**batch_dict)
        last_hidden_states = outputs.last_hidden_state # [B, L, H]
        attention_mask = batch_dict["attention_mask"]

        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            embeddings = last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            embeddings = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        return F.normalize(embeddings, p=2, dim=-1)

    def forward(self, batch):
        s1_batch_dict = {
            "input_ids": batch["s1_input_ids"],
            "attention_mask": batch["s1_attention_mask"]
        }
        s2_batch_dict = {
            "input_ids": batch["s2_input_ids"],
            "attention_mask": batch["s2_attention_mask"]
        }
        
        emb1 = self.encode(s1_batch_dict)
        emb2 = self.encode(s2_batch_dict)
        cos_sim = torch.sum(emb1 * emb2, dim=-1)  # [B]
        return cos_sim

class CosinePearsonLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CosinePearsonLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        
        correlation = F.cosine_similarity(
            pred_centered.unsqueeze(0), 
            target_centered.unsqueeze(0), 
            dim=1,
            eps=self.eps
        ).squeeze()
        
        loss = 1.0 - correlation
        return loss
    
# Load base model
base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

base_model.gradient_checkpointing_enable()  
base_model.config.use_cache = False        

base_model = get_peft_model(base_model, peft_config)

similarity_model = SimilarityModel(base_model)

# Check trainable parameters
if accelerator.is_main_process:
    similarity_model.base.print_trainable_parameters()

loss_fn = CosinePearsonLoss()

# Update only the trainable (LoRA) parameters
trainable_params = [
    p for p in similarity_model.parameters() if p.requires_grad
]
# Base learning rate
lr = 2e-5 
optimizer = AdamW(trainable_params, lr=lr)

similarity_model, optimizer, train_loader, val_loader = accelerator.prepare(
    similarity_model, optimizer, train_loader, val_loader
)

def get_peft_model_to_save(model):
    """
    Unwraps the model from DDP/FSDP (if applicable) and SimilarityModel, 
    returning the core PeftModel instance for saving.
    """
    model = accelerator.unwrap_model(model)
    
    return model.base


def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
           
            preds = model(batch)
            
            # Get predictions and labels from all processes
            preds = accelerator.gather(preds)
            labels = accelerator.gather(batch["label"])
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate Spearman correlation
    if accelerator.is_main_process:
        corr = 100 * spearmanr(all_preds, all_labels).correlation
        return corr
    else:
        # Return a placeholder for non-main processes
        return 0.0

# Training
epochs = 10
if accelerator.is_main_process:
    print(f"Starting training for {epochs} epochs...")
    print(f"Per-GPU batch size: {base_batch_size}")
    print(f"Total effective batch size: {base_batch_size * num_gpus}")
    
for epoch in range(epochs):
    similarity_model.train()
    total_loss = 0
    
    if accelerator.use_distributed and hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)
        
    for batch_idx, batch in enumerate(train_loader):
        
        preds = similarity_model(batch)
        loss = loss_fn(preds, batch["label"])

        accelerator.backward(loss)
        
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        if accelerator.is_main_process and (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Avg-Loss: {avg_loss:.4f}")

    avg_epoch_loss = total_loss / len(train_loader)
    avg_epoch_loss = accelerator.reduce(torch.tensor(avg_epoch_loss).to(device), reduction="mean").item()

    val_corr = evaluate(similarity_model, val_loader)
    train_corr = evaluate(similarity_model, train_loader)
    
    if accelerator.is_main_process:
        peft_model_instance = get_peft_model_to_save(similarity_model)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"qwen_lora_epoch_{epoch+1}")
        
        # Save only the LoRA adapter weights and config
        peft_model_instance.save_pretrained(checkpoint_path)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_epoch_loss:.4f} - Val Spearman: {val_corr:.4f} - Train Spearman: {train_corr:.4f} | LoRA saved to {checkpoint_path}")

if accelerator.is_main_process:
    peft_model_instance = get_peft_model_to_save(similarity_model)
    final_path = os.path.join(CHECKPOINT_DIR, "qwen_lora_final")
    peft_model_instance.save_pretrained(final_path)
    print(f"Training completed! Final adapter saved to {final_path}")