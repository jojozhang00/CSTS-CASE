import os
import json
import random
import warnings

import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

# Local import
from llm_encoders import LLM_EMBEDDER
random.seed(42)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.backends.cuda")
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the dataset
base_path = "./CSTS-data"

dataset = load_dataset(
    'csv', 
    data_files={
        'train': base_path + '/csts_train_reannotated.csv',
        'validation': base_path + '/csts_validation_reannotated.csv',
    }
)

validation = dataset['validation']
dataset['validation'] = validation.select(range(1983))
dataset['test'] = validation.select(range(1983, len(validation)))

# Create the embedding matrix using the LLM encoder
def llm_build_matrix(dataset, model):
    """
    We will create an embedding matrix where each row corresponds to a sentence/word embedding, encoded using the given model.
    We will also return a dictionary that maps the sentence/word to a row id, and another dictionary that does the reverse.
    
    Args:
        dataset (datasets.dataset): stores the dataset
        model (encoder): sentence encoder
        
    Returns:
        M (numpy.array): A tensor where each row is a text (sentence or word) embedding
        sent2row (dict): A dictionary where the key is the text and value is the row id holding the embedding of that text in the matrix
        row2sent (dict): A dictionary where the key is the row id and the value is the text corresponding to the embedding in the row    
    """
    sent2row = {}
    vects = []
    rowid = 0
    dropped_instances = 0
    for inst in tqdm(dataset):
        #print(rowid)
        if inst['label'] == -1: # -1 is used for the instances with invalid conditions, while test instances have label -2
            dropped_instances += 1
        else:
            c = inst['condition']
            if c not in sent2row:
                sent2row[c] = rowid
                rowid += 1
                vects.append(model.encode("", c))
            for s in [inst['sentence1'], inst['sentence2']]:
                if s not in sent2row:
                    sent2row[s] = rowid
                    rowid += 1
                    vects.append(model.encode(s, ""))
                c_s = f"{c} {s}"
                if c_s not in sent2row:
                    sent2row[c_s] = rowid
                    rowid += 1
                    vects.append(model.encode(s, c) - model.encode("", c)) # subtract c
     
    print(f"Dropped instances = {dropped_instances}")
    row2sent = {rowid: txt for txt, rowid in sent2row.items()}
    
    embed_M = np.vstack(vects)
    return embed_M, sent2row, row2sent

# save or load llm_M, llm_sent2row and llm_row2sent to/from the disk
def save_embeddings_to_file(M, sent2row, row2sent, model_name):
    base_path = '/users/sggzhan8/fastscratch'
    path = os.path.join(base_path, f'embed_matrix/{model_name}/')
    
    os.makedirs(path, exist_ok=True)
    
    print(f"Saving {model_name} to {path} with shape =", M.shape)
    M_fname = os.path.join(path, 'M.npy')
    sent2row_fname = os.path.join(path, 'sent2row.json')
    row2sent_fname = os.path.join(path, 'row2sent.json')
    
    np.save(M_fname, M)
    with open(sent2row_fname, "w") as sent2row_file:
        json.dump(sent2row, sent2row_file, indent=4)
    with open(row2sent_fname, "w") as row2sent_file:
        json.dump(row2sent, row2sent_file, indent=4)
      
def load_embeddings_from_file(M_fname, sent2row_fname, row2sent_fname):
    tmp_M = np.load(M_fname, allow_pickle=True)
    with open(sent2row_fname) as sent2row_file:
        tmp_llm_sent2row = json.load(sent2row_file)
    with open(row2sent_fname) as row2sent_file:
        tmp_llm_row2sent = json.load(row2sent_file)
    return tmp_M, tmp_llm_sent2row, tmp_llm_row2sent
      

# Create LLM embeddings and save to files
# model_name, pooling_method, encoder_type = "e5", "average", "condition"
#model_name, pooling_method, encoder_type = "SFR", "average", "condition"
#model_name, pooling_method, encoder_type = "e5", "average", "condition"
model_name, pooling_method, encoder_type = "qwen3-8B", "last", "condition"
llm_model = LLM_EMBEDDER(model_name, pooling_method, encoder_type)

#model_name = "simcse_base"
#llm_model = SimCSE("base")
# concatenate_datasets([dataset['train'],dataset['validation'], dataset['test']])

combined_dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
embed_M, sent2row, row2sent = llm_build_matrix(combined_dataset, llm_model)

save_embeddings_to_file(embed_M, sent2row, row2sent, model_name)


