import enum
import sys
import os
import csv
import re

cur_path = os.getcwd()
print(cur_path)
dat_path = os.path.join(cur_path, 'Data', 'Imdb', 'imdb_master.csv')
print('reading file: ')
print(dat_path)

raw_reviews = []
with open(dat_path) as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        raw_reviews.append(row)

print(raw_reviews[0])

test_reviews = []
test_labels = []

train_reviews = []
train_labels = []

print('splitting csv...')
cursor = 1
for index, review in enumerate(raw_reviews[cursor:], start=cursor):
    _, mode, text, rate, _ = review

    if mode == 'test':
        test_reviews.append(text)
        test_labels.append(1 if rate.startswith('pos') else 0)
    else:
        train_reviews.append(text)
        train_labels.append(1 if rate.startswith('pos') else 0)

vocab = set()
splitRegex = '[\.\s!\?,\"\'\:()\;\-\+\/\*\#`]|(<br />)'

print('creating vocabulary...')
for index, review in enumerate(train_reviews):

    words = re.split(splitRegex, review)
    for word in filter(lambda w: (not isinstance(w, type(None))) and (len(w) > 0), words):
        vocab.add(word)

vocab = list(vocab)

print('indexing vocabulary...')
word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i

print('creating input vectors...')
input_dataset = list()
for index, review in enumerate(train_reviews):
    review_vector = list()
    words = re.split(splitRegex, review)
    for word in filter(lambda w: (not isinstance(w, type(None))) and (len(w) > 0), words):
        try:
            review_vector.append(word2index[word])
        except:
            ""
    input_dataset.append(review_vector)

print('Finished successfully')