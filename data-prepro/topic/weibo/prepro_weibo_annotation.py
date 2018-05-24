# encoding:utf-8
import json
from tqdm import tqdm
import pickle as pk
from multiprocessing import Process
import sys
sys.path.append('../aichallenge')
from enuncoding import *

import thulac
thul = thulac.thulac()

def run_proc(model_name, data, id2name):
    test_data = {}
    print("Processing %s_data"%model_name)
    for i in tqdm(dataset[model_name+'_id']):
        caps = []
        # For GT
        name = id2name[i]
        count = 0
        sen = encode_utf8(data[i])
        topic = []
        for word, pos in thul.cut(''.join(sen.split())):
            if pos in ['n', 'v']:
                try:
                    topic.append(decode_any(word))
                except:
                    pass

        tmp = {}
        tmp['filename'] = name
        tmp['img_id'] = i
        tmp['cap_id'] = count
        tmp['caption'] = sen
        tmp['topic'] = topic
        count += 1
        caps.append(tmp)

        test_data[i] = caps
    print 'dump %d in %s_data'%(len(test_data), model_name)
    json.dump(test_data, open('./K_%s_annotation.json'%model_name, 'w'))

input_data = 'splits.pkl'
with open(input_data) as data_file:
    dataset = pk.load(data_file)

skip_num = 0
val_data = {}
train_data = {}
id2name = pk.load(open('id2name.pkl'))
data = pk.load(open('id2caption.pkl'))

for i in ['train', 'val', 'test']:
    p = Process(target=run_proc, args=(i, data, id2name))
    print 'Process %s will start.'%i
    p.start()
    
'''
print("Processing train_data")
for i in tqdm(dataset['train_id']):
    caps = []
    # For GT
    name = id2name[i]
    count = 0
    sen = data[i]

    topic = []
    try:
        for word, pos in thul.cut(''.join(sen.split())):
            for punc in string.punctuation:
                if punc in word:
                    word = word.replace(punc, '')
            if pos in ['n', 'v']:
                topic.append(word)
    except:
        print sen
            
    tmp = {}
    tmp['filename'] = name
    tmp['img_id'] = i
    tmp['cap_id'] = count
    tmp['caption'] = sen
    tmp['topic'] = topic
    count += 1
    caps.append(tmp)

    # print i, type(i)
    train_data[i] = caps

print 'number of skip train data: ' + str(skip_num)
# [u'info', u'images', u'licenses', u'type', u'annotations']
print 'dump %d in train_data'%len(train_data)
json.dump(train_data, open('./K_train_annotation.json', 'w'))

print("Processing val_data")
for i in tqdm(dataset['val_id']):
    caps = []
    # For GT
    name = id2name[i]
    count = 0
    sen = data[i]
    for punc in string.punctuation:
        if punc in sen:
            sen = sen.replace(punc, '')
    topic = []
    try:
        for word, pos in thul.cut(''.join(sen.split())):
            if pos in ['n', 'v']:
                topic.append(word)
    except:
        print sen
            
    tmp = {}
    tmp['filename'] = name
    tmp['img_id'] = i
    tmp['cap_id'] = count
    tmp['caption'] = sen
    tmp['topic'] = topic
    count += 1
    caps.append(tmp)

    val_data[i] = caps

print 'dump %d in val_data'%len(val_data)
json.dump(val_data, open('./K_val_annotation.json', 'w'))
'''