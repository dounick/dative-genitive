import torch
import torch.optim as optim
import numpy as np
from util import *
import os
import pickle

BERT_path = 'cached_bert'
dataset_name = 'gen'
cached_path = os.path.join(BERT_path, dataset_name)

embeddings = torch.load(os.path.join(cached_path, 'embeddings.pt'))

with open(os.path.join(cached_path, 'amper_pairs.pkl'), 'rb') as f:
    amper_pairs = pickle.load(f)
with open(os.path.join(cached_path, 'dollar_pairs.pkl'), 'rb') as f:
    dollar_pairs = pickle.load(f)

layer = 9
input_size = 768
hidden_size = 64
model= Perceptron(input_size, hidden_size)

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

classifier = Perceptron(input_size, hidden_size)
criterion = nn.BCEWithLogitsLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters())


# Create DataLoader for training
train_dataset = torch.utils.data.TensorDataset(embeddings_from_layer[:2000], 
                                               labels[:2000])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# Create DataLoader for testing

dataset_name = 'dat'
cached_path = os.path.join(BERT_path, dataset_name)

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
test_dataset = torch.utils.data.TensorDataset(embeddings_from_layer[2001:2600], 
                               labels[2001:2600])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# Call the train function
num_epochs = 100  # Define the number of training epochs
train(model, train_loader, criterion, optimizer, num_epochs)

test(model, test_loader, criterion)

