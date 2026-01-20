"""
This package contains various routines for loading and calling LLM encoders.
"""

import torch
from torch import Tensor
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.backends.cuda")
warnings.filterwarnings("ignore", category=UserWarning)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device =", device)

class LLM_EMBEDDER:
  
  def __init__(self, model_name: str, pooling_method: str = "average", encoder_type: str = "condition"):
    print(f"Loading model : {model_name}, pooling : {pooling_method}, encoder_type :{encoder_type}")
    self.model_name = model_name
    self.device = device  

    if model_name == "gte":  
      self.tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
      self.model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True, device_map="auto")
    elif model_name == "e5":
      self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
      self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct', device_map="auto")
    elif model_name == "SFR":
      self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
      self.model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral', device_map="auto")
    elif model_name == "qwen3-0.6B": 
      self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left', trust_remote_code=True, force_download=True)   
      self.model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True, force_download=True)
      self.model.to(self.device) 
    elif model_name == "qwen3-4B":
      self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B', padding_side='left', trust_remote_code=True, force_download=True)   
      self.model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B', trust_remote_code=True, force_download=True)
      self.model.to(self.device) 
    elif model_name == "qwen3-8B":
      self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left', trust_remote_code=True, force_download=True)   
      self.model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B', trust_remote_code=True, force_download=True)
      self.model.to(self.device)
    elif model_name == "nv_embed":
      self.model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True, device_map="auto")
    else:
      raise ValueError(f"Invalid model_name {model_name}.")    
    
    self.pooling_method = pooling_method
    self.encoder_type = encoder_type      
    
  def average_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
  def last_token_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

  def get_detailed_instruct(self, sentence: str, condition: str) -> str:
    # if either sentence or condition is empty, we will use STS prompt to encode the input
    task_description =  "Retrieve semantically similar texts"
    if len(sentence) == 0 and len(condition) > 0:      
      return f'Instruct : {task_description} Query : {condition}'
    if len(condition) == 0 and len(sentence) > 0:
      return f'Instruct : {task_description} Query : {sentence}'
    
    # we have both a sentence and a condition. We will use C-STS prompt
    task_description = "Retrieve semantically similar texts to a given Query, under the given Condition"
    if self.encoder_type == "sentence": # embed sentence
      return f'Instruct : {task_description} Query : {sentence}\nCondition : {condition}'
    elif self.encoder_type == "condition":  # embed condition
      return f'Instruct : {task_description} Query : {condition}\nCondition : {sentence}' 
    else:
      raise ValueError(f"Invalid encoder_type: {self.encoder_type}. Must be in [sentence, condition]")
  
  def token_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    if self.pooling_method == "last":
      return self.last_token_pool(last_hidden_states, attention_mask)
    elif self.pooling_method == "average":
      return self.average_pool(last_hidden_states, attention_mask)
    else:
      raise ValueError(f"Incorrect pooling method: {self.pooling_method}. Must be in [last, average]")
    
  def nv_encode(self, sentence, condition):    
    query_prefix = ""
    max_length = 32768
    
    if len(sentence) == 0 and len(condition) > 0:      
      query_prefix = "Retrieve semantically similar texts" 
      query_embeddings = self.model.encode([condition], instruction=query_prefix, max_length=max_length)
      return query_embeddings.squeeze(0).detach().cpu().numpy()  
    
    if len(condition) == 0 and len(sentence) > 0:
      query_prefix = "Retrieve semantically similar texts" 
      query_embeddings = self.model.encode([sentence], instruction=query_prefix, max_length=max_length)
      return query_embeddings.squeeze(0).detach().cpu().numpy()  
    
    if self.encoder_type == "sentence": 
      query_prefix = f'Retrieve semantically similar texts to a given {sentence}, under the given: {condition}' 
      query_embeddings = self.model.encode([sentence], instruction=query_prefix, max_length=max_length)
      return query_embeddings.squeeze(0).detach().cpu().numpy()  
    
    elif self.encoder_type == "condition":  
      query_prefix = f'Retrieve semantically similar texts to a given {condition}, under the given: {sentence}' 
      query_embeddings = self.model.encode([condition], instruction=query_prefix, max_length=max_length)
      return query_embeddings.squeeze(0).detach().cpu().numpy()  
    
    else:
      raise ValueError(f"Invalid encoder_type: {self.encoder_type}. Must be in [sentence, condition]")    
      return None
  
  def encode(self, sentence: str, condition: str):
    if self.model_name == "nv_embed":
        return self.nv_encode(sentence, condition)
    else:
        max_len = self.tokenizer.model_max_length
        if max_len > 1e6: 
            max_len = 2048 
        input_text = self.get_detailed_instruct(sentence, condition)
        batch_dict = self.tokenizer(input_text, max_length=max_len, padding=True, truncation=True, return_tensors='pt')
        
        batch_dict = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch_dict.items()}
        
        with torch.no_grad(): 
            outputs = self.model(**batch_dict)
        embeddings = self.token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings.squeeze(0).detach().cpu().numpy()
  
# SimCSE embeddings
class SimCSE:
      
  def __init__(self, model_name):
    
    if model_name == "large": 
      self.tokenizer = AutoTokenizer.from_pretrained(
          "princeton-nlp/sup-simcse-roberta-large",
          use_safetensors=True  
      )
      self.model = AutoModel.from_pretrained(
          "princeton-nlp/sup-simcse-roberta-large",
          use_safetensors=True 
      )
    if model_name == "base": 
      self.tokenizer = AutoTokenizer.from_pretrained(
          "princeton-nlp/sup-simcse-bert-base-uncased",
          use_safetensors=True 
      )
      self.model = AutoModel.from_pretrained(
          "princeton-nlp/sup-simcse-bert-base-uncased",
          use_safetensors=True  
      )

  def encode(self, sentence: str, condition: str) -> np.ndarray:    
    if len(sentence) == 0 and len(condition) == 0:
      raise ValueError("Both sentence and condition cannot be empty.")
    if len(sentence) > 0 and len(condition) > 0:
      text = f"{sentence} {condition}"
    elif len(sentence) == 0:
      text = condition
    elif len(condition) == 0:
      text = sentence      
    input = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    
    device = next(self.model.parameters()).device
    input = {k: v.to(device) for k, v in input.items()}
    
    with torch.no_grad():
      embeddings = self.model(**input, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings[0]

def test():
    llm_model = LLM_EMBEDDER(model_name="qwen3-8B", pooling_method="last", encoder_type="condition")
    instances = [("Young woman in orange dress about to serve in tennis game, on blue court with green sides.",
              "A girl playing tennis wears a gray uniform and holds her black racket behind her.", 
              "The color of the dress.", 1),
             ("Young woman in orange dress about to serve in tennis game, on blue court with green sides.",
              "A girl playing tennis wears a gray uniform and holds her black racket behind her.",
              "The name of the game.", 5)]
    for i in range(0, 2):
        (s1, s2, c, label) = instances[i]
        embd_s1_c = llm_model.encode(s1, c) - llm_model.encode("", c)
        embd_s2_c = llm_model.encode(s2, c) - llm_model.encode("", c)
        sim = cosine_similarity([embd_s1_c], [embd_s2_c])[0][0]
        print(sim)
            
if __name__ == "__main__":
    test()
