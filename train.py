# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification

# specify GPU
device = torch.device("cpu")

""" Load Dataset and Split train dataset into train, validation and test sets"""


def clean_text(x):
    pattern = r'\([^()]*\)'
    text = re.sub(pattern, '', x)
    return text


def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x


train_text = []
train_labels = []
val_text = []
val_labels = []

with open('data/train.jsonl', 'rb') as infile:
    for line in infile.readlines():
        entry = json.loads(line)
        text = clean_numbers(clean_text(entry['response'] + ' ' + ' '.join(entry['context'])))
        train_text.append(text)
        if entry['label'] == 'SARCASM':
            train_labels.append(1)
        else:
            train_labels.append(0)

train_text, val_text, train_labels, val_labels = train_test_split(train_text, train_labels,
                                                                    random_state=2020,
                                                                    test_size=0.1,
                                                                    stratify=train_labels)

test_text = []
test_labels = []
test_ids = []
with open('data/test.jsonl', 'rb') as infile:
    for line in infile.readlines():
        entry = json.loads(line)
        text = clean_numbers(clean_text(entry['response'] + ' ' + ' '.join(entry['context'])))
        test_text.append(text)
        test_labels.append(1)
        test_ids.append(entry['id'])


"""# Import BERT Model and BERT Tokenizer"""

# import BERT-base pretrained model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

"""# Tokenization"""

max_seq_len = 40

# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text,
    max_length=max_seq_len,
    padding=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text,
    max_length=max_seq_len,
    padding=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text,
    max_length=max_seq_len,
    padding=True,
    truncation=True,
    return_token_type_ids=False
)

"""# Convert Integer Sequences to Tensors"""

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels)

# for validation set
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels)

# for test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels)

"""# Create DataLoaders"""

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# define a batch size
batch_size = 32

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

# push the model to GPU
model = model.to(device)

# optimizer from hugging face transformers
from transformers import AdamW

# define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

"""# Find Class Weights"""

from sklearn.utils.class_weight import compute_class_weight

# compute the class weights
class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)

# convert class weights to tensor
weights = torch.tensor(class_wts, dtype=torch.float)
weights = weights.to(device)

# loss function
cross_entropy = nn.NLLLoss(weight=weights)

# number of training epochs
epochs = 5

"""# Fine-Tune BERT"""


# function to train the model
def train():
    print("\nTraining Process Start...")
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 40 batches.
        if step % 40 == 0 and not step == 0:
            print('\nBatch round {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        (loss, preds) = model(sent_id,
                              token_type_ids=None,
                              attention_mask=mask,
                              labels=labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds


# function for evaluating the model
def evaluate():
    print("\nEvaluating Process Start...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Report progress.
            print('  Batch Time {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # # model predictions
            # preds = model(sent_id, mask)
            #
            # # compute the validation loss between actual and predicted values
            # loss = cross_entropy(preds, labels)
            (loss, preds) = model(sent_id,
                                   token_type_ids=None,
                                   attention_mask=mask,
                                   labels=labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


"""# Start Model Training"""

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses = []
valid_losses = []
# for each epoch
for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    # train model
    train_loss, _ = train()

    # evaluate model
    valid_loss, _ = evaluate()

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')

    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

"""# Load Saved Model"""

# load weights of best model
path = 'saved_weights.pt'
model.load_state_dict(torch.load(path))

"""# Get Predictions for Test Data"""

# get predictions for test data
print("Predication Start...")
with torch.no_grad():
    preds = model(test_seq.to(device), attention_mask=test_mask.to(device))
    preds = preds[0].detach().cpu().numpy()
print(preds)

# choose and save data following the requirements
preds = np.argmax(preds, axis=1)
with open('answer.txt', 'w+') as f:
    for i, id in enumerate(test_ids):
        if preds[i] == 1:
            f.write(id + ',' + 'SARCASM' + '\n')
        else:
            f.write(id + ',' + 'NOT_SARCASM' + '\n')
