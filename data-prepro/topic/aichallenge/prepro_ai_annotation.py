#coding:utf-8
from multiprocessing import Process
import json
import sys
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from random import shuffle, seed
import pickle as pk

def run_proc(name, data, id2name):
    print("Processing %s_data"%name)
    test_data = {}
    for i in tqdm(dataset[name+'_id']):
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
                    if pos in ['n', 'v']:
                        topic.append(word)
                tmp['topic'] = topic
                count += 1
                caps.append(tmp)
            test_data[i] = caps
    print 'dump %d in %s_data'%(len(test_data), name)
    json.dump(test_data, open('./K_%s_annotation.json'%name, 'w'))

input_data = 'splits.pkl'
with open(input_data) as data_file:
    dataset = pk.load(data_file)

phase = sys.argv[1]
skip_num = 0
val_data = {}
train_data = {}
id2name = pk.load(open('id2name.pkl'))
data = pk.load(open('filename2caption.pkl'))

import thulac
thul = thulac.thulac()

for i in ['train', 'val', 'test']:
    p = Process(target=run_proc, args=(i, data, id2name))
    print 'Process %s will start.'%i
    p.start()
    # print 'Process %s end.'%i

'''
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
                if pos in ['n', 'v']:
                    topic.append(word)
            tmp['topic'] = topic
            count += 1
            caps.append(tmp)

        val_data[i] = caps
print 'dump %d in val_data'%len(val_data)
json.dump(val_data, open('mscoco_data/K_val_annotation.json', 'w'))
'''
