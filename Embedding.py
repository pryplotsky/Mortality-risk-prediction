#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:26:13 2022

@author: discharge
"""


from transformers import BertTokenizer, BertModel
import numpy as np
import torch
import pandas as pd
import pickle
import time
#Define global variables
cwd=r'/home/discharge/Desktop/Bohdan'
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased', proxies={"http":"http://proxy.charite.de:8080","https": "http://proxy.charite.de:8080"})#Get BERT tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model = BertModel.from_pretrained('bert-base-german-cased', proxies={"http":"http://proxy.charite.de:8080","https": "http://proxy.charite.de:8080"}).to(device)#Get pretrained BERT model
#Define functions
'''
Get document embedding using pretrained German BERT model and  German BERT tokenizer
Args:
    document (str): textual document which has to be transformed 
'''  
def get_document_embedding(document):
  with torch.no_grad():
    inputs = tokenizer(document, return_tensors="pt", max_length=512, truncation=True)#Tokenize the dataset, truncate when passed `max_length`,and pad with 0's when less than `max_length`,save as pytorch  return_tensors,
    inputs.to(device)#To GPU if available
    outputs = bert_model(**inputs)#Get BERT embedding
    document_embedding = outputs.last_hidden_state[0][0]#Select CLS token (always first one)
    document_embedding = document_embedding.unsqueeze_(0)#Return horisontal position of tensor, underscore means no new memory is being allocated by doing the operation, which in general increase performance
    return document_embedding
'''
Get hospitalization embedding using embedding from devied above function get_day_embedding
Args:
    document_list (list): list of documents for one hospitalization
'''
def get_day_embedding(document_list):
    document_embeddings = get_document_embedding(document_list[0]).unsqueeze_(0)#Store first embedding
    for document in document_list[1:]:
      document_embeddings = torch.cat([document_embeddings, get_document_embedding(document).unsqueeze_(0)],0)#Concatenate all embeddings for one hospitalization
    print(document_embeddings.shape)
    day_embedding = torch.mean(document_embeddings, dim=0).to(device)# Calculate mean for one hospitalization 
    print(day_embedding.shape)
    return day_embedding
'''
Pass in slice from input, return tensor of embeddings [SEQ_LEN, EMBEDDING_DIM]
Args:
    sequence (numeric): input sequence (embedding)
'''
def embeddings_to_tensor(sequence): 
  # pass in slice from dataset, return tensor of embeddings [SEQ_LEN, EMBEDDING_DIM]
  sequence_tensor = sequence[0:1][0]#Store first embedding
  for i in range(1,len(sequence)):
    sequence_tensor = torch.cat([sequence_tensor, sequence[i]],0)#Concatenate all embeddings for one hospitalization 
  sequence_tensor.to(device)#To GPU if available
  return sequence_tensor


#Read dataframe
with open( cwd + '/' +  'df' + '.pkl' ,'rb') as path_name:# load df, 'rb' specifies 'read'
  dataframe = pickle.load(path_name)
#Define sets of variables to transform and aggregate as a list
X_set=dataframe.copy()
X_set=X_set.loc[X_set['text'].isna()==False] 
X_set_2 = X_set.groupby('Fallpseudonym2')[['text']].agg({"text":list})
#Get hospitalization embedding using embedding from devied above function get_day_embedding
start_time = time.time()
to_rem=np.array(X_set_2.text)
for i in range(0,len(to_rem)):
       to_rem[i]=get_day_embedding(to_rem[i]) 
print("--- %s seconds ---" % (time.time() - start_time))
#Save hospitalization embedding as dataframe
start_time = time.time() 
df_bert_train=pd.DataFrame(embeddings_to_tensor(to_rem).cpu().numpy())
print("--- %s seconds ---" % (time.time() - start_time))
#Save text embedding
df_bert_train=df_bert_train.set_index(X_set_2.index)
with open(cwd +"\df_bert_train.pkl", 'wb') as file:  
    pickle.dump(df_bert_train , file)
with open(cwd +"\df_bert_train.pkl", 'rb') as file:  
    df_bert_train = pickle.load(file)   