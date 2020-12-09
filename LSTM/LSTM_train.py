import argparse
import re
import torch
import numpy as np
from torch import nn, optim
from torch.utils import data
from torch.utils.data import DataLoader
from LSTM_model import Model
from LSTM_dataset import Dataset

training_on = False

def predict(dataset: Dataset, model, text, next_words=100):
    splitRegex = '[\s\"\*\#\:()\;\-\+\/]'
    words = re.split(splitRegex, text)
    words = list(filter(lambda word: word is not None and len(word) > 0, words))
    model.eval()

    state_h, state_c = model.init_state(len(words))

    words_corrected = []
    for w in words:
        if w in dataset.word_to_index:
            words_corrected.append(w)
        else:
            words_corrected.append('frog')

    can_continue = True
    iter = 0
    complete_prob = 0
    sent = 1

    while can_continue:

        indexed_words = [[dataset.word_to_index[w] for w in words_corrected[iter:]]]
        x = torch.tensor(indexed_words)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)


        word_pred = dataset.index_to_word[word_index]

        include_word = True
        if word_pred[0].isupper():
            if words_corrected[-1].lower() in ('and', 'if', 'the', 'a', 'to', 'from', 'on', 'in', 'for'):
                complete_prob = min(1, complete_prob + 0.01)
            else:
                complete_prob = min(1, sent / 10)
                sent += 1
                include_word = False
        elif word_pred.endswith(('.', '?', '!', ';', '^')):
            complete_prob = min(1, sent / 3)
            sent += 1
        else:
            complete_prob = min(1, complete_prob + 0.03)

        rand = np.random.random()
        can_continue = rand >= complete_prob
        
        if can_continue or include_word: 
            words_corrected.append(word_pred)
        
        if not can_continue and not include_word:
            words_corrected.append('[ {} ]'.format(word_pred))

        iter += 1

    return words_corrected

def test(dataset, model):
    things = [
        'king',
        'programmer',
        'lazy rat',
        'cow',
        'girlfiend',
        'pigeon',
        'Facebook',
        'Trump',
        'Putin',
        'witch',
    ]

    for i in range(len(things)):
        words = predict(dataset, model, text='{} enters bus and'.format(things[i]), next_words=20)
        print(' '.join(words))

def train(dataset, model, args):
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)
        batches = int(len(dataloader.dataset)/dataloader.batch_size)
        for batch, (x, y) in enumerate(dataloader):

            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'of': batches, 'loss': loss.item() })

            if batch % 100 == 1:
                model.save()
                test()

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=5)
args = parser.parse_args()

dataset = Dataset(args)
model = Model(dataset)

model.load()

if training_on:
    train(dataset, model, args)
    model.save()

test(dataset, model)
