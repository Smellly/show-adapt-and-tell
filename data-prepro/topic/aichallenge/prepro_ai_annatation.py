#coding:utf-8
import json
import string
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from random import shuffle, seed
import pickle as pk
import pdb

input_data = 'splits.pkl'
with open(input_data) as data_file:
    dataset = pk.load(data_file)

skip_num = 0
val_data = {}
test_data = {}
train_data = {}
val_dataset = []
test_dataset = []
counter = 0
id2name = pk.load(open('id2name.pkl'))
data = pk.load(open('filename2caption.pkl'))

import thulac
thul = thulac.thulac()

print("Processing test_data")
for i in tqdm(dataset['test_id']):
        caps = []
        # For GT
        name = id2name[i]
        count = 0
        for cap in data[name]:
            tmp = {}
            tmp['filename'] = name
            tmp['img_id'] = i
            tmp['cap_id'] = count 
            tmp['caption'] = cap
            topic = []            
            for word, pos in thul.cut(''.join(cap.split())):
                for punc in string.punctuation:
                    if punc in word:
                       word = word.replace(punc, '') 
                if pos in ['n', 'v']:
                    topic.append(word)
            tmp['topic'] = topic
            count += 1
            caps.append(tmp)

        test_data[i] = caps

print 'dump %d in test_data'%len(test_data)
#print test_data
json.dump(test_data, open('./K_test_annotation.json', 'w'))
# pk.dump(test_data, open('mscoco_data/K_test_annotation.pkl', 'w'))

print("Processing train_data")
for i in tqdm(dataset['train_id']):
        caps = []
        # For GT
        name = id2name[i]
        count = 0
        for cap in data[name]:
            tmp = {}
            tmp['filename'] = name
            tmp['img_id'] = i
            tmp['cap_id'] = count
            tmp['caption'] = cap
            topic = []            
            for word, pos in thul.cut(''.join(cap.split())):
                for punc in string.punctuation:
                    if punc in word:
                       word = word.replace(punc, '') 
                if pos in ['n', 'v']:
                    topic.append(word)
            tmp['topic'] = topic
            count += 1
            caps.append(tmp)
        # print i, type(i)
        train_data[i] = caps
print 'number of skip train data: ' + str(skip_num)
print 'dump %d in train_data'%len(train_data)
json.dump(train_data, open('mscoco_data/K_train_annotation.json', 'w'))
# pk.dump(train_data, open('mscoco_data/K_train_annotation.pkl', 'w'))

print("Processing val_data")
for i in tqdm(dataset['val_id']):
        caps = []
        # For GT
        name = id2name[i]
        count = 0
        for cap in data[name]:            
            tmp = {}
            tmp['filename'] = name
            tmp['img_id'] = i
            tmp['cap_id'] = count
            tmp['caption'] = cap
            topic = []            
            for word, pos in thul.cut(''.join(cap.split())):
                for punc in string.punctuation:
                    if punc in word:
                       word = word.replace(punc, '') 
                if pos in ['n', 'v']:
                    topic.append(word)
            tmp['topic'] = topic
            count += 1
            caps.append(tmp)

        val_data[i] = caps
# pk.dump(val_data, open('mscoco_data/K_val_annotation.pkl', 'w'))
print 'dump %d in val_data'%len(val_data)
json.dump(val_data, open('mscoco_data/K_val_annotation.json', 'w'))
