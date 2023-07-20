import torch
import numpy as np
from util import *
import os
import pickle

#loading models and datasets from layer
BERT_path = 'cached_bert'
model_type = 'gen'
dataset_type = 'dat'
layer = 6
cached_path = os.path.join(BERT_path, dataset_type)

embeddings = torch.load(os.path.join(cached_path, 'embeddings.pt'))

with open(os.path.join(cached_path, 'amper_pairs.pkl'), 'rb') as f:
    amper_pairs = pickle.load(f)
with open(os.path.join(cached_path, 'dollar_pairs.pkl'), 'rb') as f:
    dollar_pairs = pickle.load(f)
embeddings_from_layer = []
labels = []
for i in range(len(embeddings)):
    dollar_pair = dollar_pairs[i]
    embeddings_from_layer.append(embeddings[i][dollar_pair[0]][layer])
    labels.append(0)
    amper_pair = amper_pairs[i]
    embeddings_from_layer.append(embeddings[i][amper_pair[0]][layer])
    labels.append(1)
embeddings_from_layer = torch.stack(embeddings_from_layer)
labels = torch.tensor(labels, dtype=torch.float32)

#even range is for possessor/recipient, odd range is for possessum/theme
test_dataset = torch.utils.data.TensorDataset(embeddings_from_layer[range(2000,2700,2)], 
                               labels[range(2000,2700,2)])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

input_size = 768
hidden_size = 64
model= Perceptron(input_size, hidden_size)

#loading in model
model_path = 'models'
model.load_state_dict(torch.load(os.path.join(model_path, model_type, 
                                              str(layer) + '.pth')))
criterion = nn.BCEWithLogitsLoss()
test(model, test_loader, criterion)
