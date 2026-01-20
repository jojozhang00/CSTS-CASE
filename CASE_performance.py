import json
import random
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets

random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset
base_path = "./CSTS-data"
dataset = load_dataset(
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

# load saved embedding matrix
save_path = "./"
llm_fname = save_path + "M.npy"  
llm_M = np.load(llm_fname, allow_pickle=True)
print("llm_M shape:", llm_M.shape)
# read json data
sent2row_fname = save_path + "sent2row.json"
row2sent_fname = save_path + "row2sent.json"

with open(sent2row_fname, "r") as f:
    llm_sent2row = json.load(f)

with open(row2sent_fname, "r") as f:
    llm_row2sent = json.load(f)

print(f"sent2row has {len(llm_sent2row)} records")
print(f"row2sent has {len(llm_row2sent)} records")


# evaluate function
def mahala_sim(x, y, M=None):
    if M is None:
        return cosine_similarity([x], [y])[0][0]
    else:
        delta = x - y
        return np.exp(-np.dot(np.dot(delta.T, M), delta))

def spearman_corr(preds, targets):
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    return 100 * spearmanr(preds, targets).correlation

def evaluate(embed_M, llm_sent2row, M=None):
    df = pd.DataFrame(columns=['sentence1', 'sentence2', 'condition', 'label', 'similarity'])    
    missing_count = 0
    found_count = 0

    for inst in dataset['test']:
        if inst['label'] != -1:
            try:
                key1 = f"{inst['condition']} {inst['sentence1']}"
                key2 = f"{inst['condition']} {inst['sentence2']}"
                
                row_id1 = llm_sent2row[key1]
                row_id2 = llm_sent2row[key2]
                
                s1_c = embed_M[row_id1]
                s2_c = embed_M[row_id2]
                
                sim = mahala_sim(s1_c, s2_c, M)
                
                df.loc[len(df)] = {
                    'sentence1': inst['sentence1'],
                    'sentence2': inst['sentence2'],
                    'condition': inst['condition'],
                    'label': inst['label'],
                    'similarity': sim
                }
                found_count += 1
                
            except KeyError:
                missing_count += 1
                continue

    if found_count == 0:
        print("Error: No valid instances found in the embedding matrix for evaluation!")
        return {"spearman": 0, "pval": 0, "df": df}

    spearman, pval = spearmanr(df['label'], df['similarity'])    
        
    return {
        "spearman": spearman * 100,
        "pval": pval,
        "df": df
    }

res = evaluate(llm_M, llm_sent2row)
print(f"Spearman = {res['spearman']} ({res['pval']}) before supervised projection")

def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation: {name}")

class OneLayerNonLinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0, device="cuda"):
        super(OneLayerNonLinearProjection, self).__init__()
        self.device = device
        
        # Linear -> LeakyReLU -> Dropout
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = get_activation("leakyrelu")  
        self.dropout = nn.Dropout(dropout_prob)
        
        self.to(device)
        
    def forward(self, embd_1, embd_2):
        embd_1 = embd_1.to(self.device)
        embd_2 = embd_2.to(self.device)
        
        proj_1 = self.projection(embd_1)
        proj_1 = self.activation(proj_1)
        proj_1 = self.dropout(proj_1)
        
        proj_2 = self.projection(embd_2)
        proj_2 = self.activation(proj_2)
        proj_2 = self.dropout(proj_2)

        return proj_1, proj_2
    
    def fit(self, M):
        self.eval() 
        
        embedding_tensor = M if isinstance(M, torch.Tensor) else torch.tensor(M, dtype=torch.float32)
        embedding_tensor = embedding_tensor.to(self.device)
        
        with torch.no_grad(): 
            proj_M = self.activation(self.projection(embedding_tensor))
            
        self.train()
        return proj_M.cpu().numpy()

# Train a Siamese bi-encoder for projecting the sentence embeddings to a lower-dimensional space and then measure their similarity
def supervised_projection(embd_M, sent2row, k, dataset, num_epochs=20, batch_size=50, dropout_prob=0):
    proj_model = OneLayerNonLinearProjection(embd_M.shape[1], k, dropout_prob).to(device)
    
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(proj_model.parameters(), lr=1e-3)
    
    train_dataset = concatenate_datasets([dataset['train'], dataset['validation']])
    
    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    embedding_tensor = torch.tensor(embd_M, dtype=torch.float32)
    embedding_layer = nn.Embedding.from_pretrained(embedding_tensor, freeze=True).to(device)

    print(f"Training started. #instances: {len(train_dataset)}, #batches: {len(val_dataloader)}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        spearman_scores = []
        proj_model.train() 
        
        for batch in val_dataloader:
            sent1_ids = []
            sent2_ids = []
            ratings = []
            
            batch_iter = zip(batch['sentence1'], batch['sentence2'], batch['condition'], batch['label'])
            
            for (sent1, sent2, cond, label) in batch_iter:
                try:
                    lbl = float(label)
                except:
                    continue 

                if lbl != -1:
                    key1 = f"{cond} {sent1}"
                    key2 = f"{cond} {sent2}"
                    
                    try:
                        id1 = sent2row[key1]
                        id2 = sent2row[key2]
                        
                        sent1_ids.append(id1)
                        sent2_ids.append(id2)
                        ratings.append(lbl / 5.0)
                    except KeyError:
                        continue

            if len(ratings) == 0:
                continue

            b_sent1 = torch.tensor(sent1_ids, dtype=torch.long).to(device)
            b_sent2 = torch.tensor(sent2_ids, dtype=torch.long).to(device)
            b_ratings = torch.tensor(ratings, dtype=torch.float32).to(device)
            
            # Forward pass
            s1_vec = embedding_layer(b_sent1)
            s2_vec = embedding_layer(b_sent2)
            
            s1_proj, s2_proj = proj_model(s1_vec, s2_vec)
            
            cos_sim = nn.functional.cosine_similarity(s1_proj, s2_proj)
            
            loss = mse_loss(cos_sim, b_ratings)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        
            
            batch_score = spearman_corr(cos_sim, b_ratings)
            if not np.isnan(batch_score):
                spearman_scores.append(batch_score)
            
        avg_spearman = sum(spearman_scores) / len(spearman_scores) if spearman_scores else 0
        
        # evaluation on the test set
        proj_model.eval()
        with torch.no_grad():
             projected_M = proj_model.fit(embd_M)
             
        res = evaluate(projected_M, sent2row)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train-r: {avg_spearman:.4f}, Test-r: {res['spearman']:.4f}")  

    return proj_model

proj_model = supervised_projection(llm_M, llm_sent2row, 512, dataset, num_epochs=30, batch_size=512, dropout_prob=0.15)
