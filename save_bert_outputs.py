from transformers import BertTokenizer, BertModel
import pickle 
import pandas as pd
from tqdm import tqdm
import os
from util import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

dataset_path = './labeled_datasets'
gen = pd.read_csv(os.path.join(dataset_path, 'NZ_gen_attempt.csv'))
sentences = gen['Line']
embeddings = []
dollar_pairs = []
amper_pairs = []

for sentence in tqdm(sentences):
    _, tokens_tensor, segments_tensor, dpair, apair = sentence_to_input(sentence, tokenizer)
    list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensor, model)
    embeddings.append((list_token_embeddings))
    dollar_pairs.append(dpair)
    amper_pairs.append(apair)

BERT_path = './cached_bert/gen'
torch.save(embeddings, os.path.join(BERT_path, "embeddings.pt"))
with open(os.path.join(BERT_path, "dollar_pairs.pkl"), "wb") as f:
    pickle.dump(dollar_pairs, f)
with open(os.path.join(BERT_path, "amper_pairs.pkl"), "wb") as f:
    pickle.dump(amper_pairs, f)
