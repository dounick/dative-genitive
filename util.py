import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import re

def cos_sim(A, B):
    return np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

def get_bert_embeddings(tokens_tensor, segments_tensor, model):
    '''
    bert embeddings for a whole sentence from a given layer
    '''
    # gradient calculation id disabled
    with torch.no_grad():
      # obtain hidden states
      outputs = model(tokens_tensor, segments_tensor)
      hidden_states = outputs[2]    # concatenate the tensors for all layers
    # use "stack" to create new dimension in tensor
    token_embeddings = torch.stack(hidden_states, dim=0)    # remove dimension 1, the "batches"
    token_embeddings = torch.squeeze(token_embeddings, dim=1)    # swap dimensions 0 and 1 so we can loop over tokens
    token_embeddings = token_embeddings.permute(1,0,2)    # intialized list to store embeddings
    # "token_embeddings" is a [Y x 12 x 768] tensor
    # where Y is the number of tokens in the sentence    # loop over tokens in sentence
    # each token in token_embeddings is a [12 x 768] tensor    # sum the vectors from the last four layers
    return token_embeddings

def sentence_to_input(text, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    tokenized_unannotated = []

    dpair1, dpair2 = -1,-1
    apair1, apair2 = -1,-1
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == '$':
            if dpair1 == -1:
                dpair1 = i+1
            else: 
                dpair2 = i-1
        elif tokenized_text[i] == '&':
            if apair1 == -1:
                apair1 = i+1
            else: 
                apair2 = i-1
        else:
            tokenized_unannotated.append(tokenized_text[i])
    if dpair2 < apair1:
        dpair1 -= 1;    dpair2 -= 1;    apair1 -= 3;    apair2 -= 3
    else:
        dpair1 -= 3;    dpair2 -= 3;    apair1 -= 1;    apair2 -= 1
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_unannotated)
    segments_ids = [1]*len(tokenized_unannotated)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])
    return tokenized_unannotated, tokens_tensor, segments_tensor, (dpair1,dpair2), (apair1,apair2)

def modify_dative_sentence(sentence, construction, verb, recipient, theme):
    construction_pattern = re.compile(re.escape(construction))
    match = construction_pattern.search(sentence)
    if match:
        construction_start, construction_end = match.span()
    else:
        return -1
    
    pattern = fr"{verb}(?:\s+{recipient})?\s+{theme}"
    match = re.search(pattern, construction)
    
    if match:
        start, end = match.span()
        modified_sentence = sentence[:construction_start + start] + f"{verb} &{theme}& to ${recipient}$" + sentence[construction_end:]
        return modified_sentence
    else:
        return -1

def modify_genitive_sentence(sentence, possessor, possessum, type):
    marked_sentence = sentence
    if type == 'S':
        possessor_pattern = re.compile(rf"\b{re.escape(possessor)}\b")
        possessum_pattern = re.compile(rf"\b{re.escape(possessum)}\b")

        # Find all occurrences of possessor and possessum in the sentence
        possessor_matches = [(m.start(), m.end()) for m in possessor_pattern.finditer(sentence)]
        possessum_matches = [(m.start(), m.end()) for m in possessum_pattern.finditer(sentence)]
        for possessor_start, possessor_end in possessor_matches:
            for possessum_start, possessum_end in possessum_matches:
                if possessum_start > possessor_end and possessum_start - possessor_end <= 3:
                    # Mark the possessor and possessum in the sentence
                    marked_sentence = marked_sentence[:possessor_start] + f"${possessor}$" + marked_sentence[possessor_end:possessum_start] + f"&{possessum}&" + marked_sentence[possessum_end:]
                    break
        # Find the match in the sentence
        if marked_sentence != sentence:
            return marked_sentence
        else:
            return -1
    elif type == 'OF':
        pattern = re.compile(rf"\b({re.escape(possessum)})\s+of\s+({re.escape(possessor)})\b")
        # Find all occurrences of the pattern in the sentence
        matches = re.findall(pattern, sentence)
        # Mark the possessor and possessum in the sentence
        for match in matches:
            possessum_marked = f"&{match[0]}&"
            possessor_marked = f"${match[1]}$"
            marked_sentence = marked_sentence.replace(match[0], possessum_marked, 1).replace(match[1], possessor_marked, 1)
            break

    if marked_sentence != sentence:
        return marked_sentence
    else: 
        return -1        

def create_train_NZ(gen_path, dat_path, directory):
    train_sentences = []
    train_indices = []
    train_labels = []
    gens = pd.read_csv(gen_path)
    dat = pd.read_csv(dat_path)
    
    #dative is treated as default (0) and genitive is alternative (1)
    for i in range(2000):
        ind = find_word_index(gens['Line'][i], gens['Construction'][i], gens['rec'][i])
        if ind != None:
            train_indices.append(find_word_index(gens['Line'][i], gens['Construction'][i], gens['rec'][i]))
        else:
            continue
        train_sentences.append(gens['Line'][i])
        train_labels.append(gens['morph_case'][i]=='dat')
    for i in range(1000):
        ind = find_word_index(dat['Line'][i], dat['Construction'][i], dat['rec'][i])
        if ind != None:
            train_indices.append(find_word_index(dat['Line'][i], dat['Construction'][i], dat['rec'][i]))
        else:
            continue
        train_sentences.append(dat['Line'][i])
        train_labels.append(dat['morph_case'][i] == 'dat')
    #shuffle_train = np.random.permutation(len(train_sentences))
    #train_sentences = train_sentences[shuffle_train]
    #train_indices = train_indices[shuffle_train]
    #train_labels = train_labels[shuffle_train]
    if(len(train_indices) != len(train_sentences)):
        print('NO')
        return

    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(os.path.join(directory, "train_sentences_NZ.npy"), np.array(train_sentences))
    np.save(os.path.join(directory, "train_indices_NZ.npy"), np.array(train_indices))
    np.save(os.path.join(directory, "train_labels_NZ.npy"), np.array(train_labels))

def create_test_NZ(gen_path, dat_path, directory):
    test_sentences = []
    test_indices = []
    test_labels = []
    gens = pd.read_csv(gen_path)
    dat = pd.read_csv(dat_path)
    
    #dative is treated as default (0) and genitive is alternative (1)
    for i in range(2001,2401):
        ind = find_word_index(gens['Line'][i], gens['Construction'][i], gens['rec'][i])
        if ind != None:
            test_indices.append(find_word_index(gens['Line'][i], gens['Construction'][i], gens['rec'][i]))
        else:
            continue
        test_sentences.append(gens['Line'][i])
        test_labels.append(gens['morph_case'][i]=='dat')
    for i in range(1001,1201):
        ind = find_word_index(dat['Line'][i], dat['Construction'][i], dat['rec'][i])
        if ind != None:
            test_indices.append(find_word_index(dat['Line'][i], dat['Construction'][i], dat['rec'][i]))
        else:
            continue
        test_sentences.append(dat['Line'][i])
        test_labels.append(dat['morph_case'][i] == 'dat')
    #shuffle_train = np.random.permutation(len(test_sentences))
    #test_sentences = test_sentences[shuffle_train]
    #test_indices = test_indices[shuffle_train]
    #test_labels = test_labels[shuffle_train]
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(os.path.join(directory, "test_sentences_NZ.npy"), np.array(test_sentences))
    np.save(os.path.join(directory, "test_indices_NZ.npy"), np.array(test_indices))
    np.save(os.path.join(directory, "test_labels_NZ.npy"), np.array(test_labels))

#given a (sentence) and a (context) and a (word) in the context, return index of word relative to sentence     
def find_word_index(sentence, context, word):
# Split the sentence into words
    words = sentence.split()

    # Find the starting position of the substring in the sentence
    start_position = sentence.find(context)
    skip = len(sentence[0:start_position].split())
    if start_position != -1:
        # Find the position of the word relative to the sentence
        try:
            word_position = words[skip:].index(word.split()[0])
        except: 
            if(word[-1] != 's' or word == 'Jesus'):
                word_position = words[skip:].index(word + '\'s')
            else:
                word_position = words[skip:].index(word + '\'')
        if word_position != -1:
            # Calculate the absolute position of the word in the sentence
            absolute_position = skip + word_position
            return absolute_position
    else:
        print("The given substring is not found in the sentence.")

def remove_symbols(word):
    return re.compile('[^a-zA-Z]')

#input: array of sentences, array of indices, choice of hidden layer
#output: array of embeddings for target words
def get_word_embeddings(sentences, word_indices, layer_index, tokenizer, model, directory, file_name):
    model.eval()
    embeddings = []
    max_sequence_length = 512
    # Tokenize the sentence
    for i in range(len(sentences)):
        sentence = sentences[i]
        word_index = word_indices[i]
        tokens = tokenizer.tokenize(sentence)

        # Map the tokenized sentence to input IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Find the start position of each token in the input IDs
        token_start_positions = []
        for i, token in enumerate(tokens):
            if i == 0:
                token_start_positions.append(0)
            else:
                token_start_positions.append(token_start_positions[i-1] + len(tokenizer.tokenize(token)))

        # Find the start and end positions of the word's tokens in the input IDs
        start_pos = token_start_positions[word_index]
        if word_index + 1 < len(token_start_positions):
            end_pos = token_start_positions[word_index + 1]
        else:
            end_pos = len(input_ids)

        # Convert the input IDs to a tensor
        input_ids_tensor = torch.tensor([input_ids])[:, :max_sequence_length]

        # Feed the input IDs through the model
        outputs = model(input_ids_tensor)
        hiddens = outputs.hidden_states

        # Retrieve the output embeddings from the specified layer
        layer_embeddings = hiddens[layer_index]

        # Extract the embeddings corresponding to the word as the average of its tokens
        word_embedding = layer_embeddings[0, start_pos:end_pos].mean(dim=0)
        embeddings.append(word_embedding.detach().numpy())

    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(os.path.join(directory, file_name), np.array(embeddings))


# Define the model
class Perceptron(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Perceptron, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Dropout(.1)
        )
        
    def forward(self, x):
        return self.main(x)

# Define the training loop
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            #print(outputs)
            loss = criterion(outputs, labels.reshape(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted[:,0] == labels).sum().item()
            total += labels.size(0)
        if epoch % 10 == 9:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {correct / total}")

    print("Training complete!")

# Define a function for testing the classifier
def test(classifier, test_loader, criterion):
    classifier.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = classifier(inputs)
            loss = criterion(outputs, labels.reshape(-1,1))
            total_loss += loss.item()
            
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted[:,0] == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    average_loss = total_loss / len(test_loader)
    
    print(f"Testing Loss: {average_loss}, Accuracy: {accuracy}")

class ShuffledDataset(Dataset):
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels
    
    def __getitem__(self, index):
        vector = self.vectors[index]
        label = self.labels[index]
        return vector, label
    
    def __len__(self):
        return len(self.vectors)
