# encoding: utf-8
import numpy as np
import os
import cPickle
from tqdm import tqdm
from chardet import detect

# generate name2id & id2name dictionary
name_id_path = './AIchallengeSet.txt'
name_id = open(name_id_path).read().splitlines()
name2id = {}
id2name = {}
for ind, i in enumerate(name_id):
    name2id[i] = ind
    id2name[ind] = i

cPickle.dump(name2id, open('name2id.pkl', 'wb'))
cPickle.dump(id2name, open('id2name.pkl', 'wb'))

# generate id2caption dictionary for all images
# please download caption data on https://github.com/reedscot/cvpr2016. 
# CUB_CVPR16 will be created after unzipping. 
caption_path = './seg.AIchallenge.caption.txt'
rf_cap = open(caption_path, 'r')
captions = rf_cap.readlines()
id2caption = {}
filename2caption = {}
for i in tqdm(xrange(len(captions))):
    #print captions[i].strip().split('####')[1]
    name = captions[i].strip().split('#')[0]
    cap = captions[i].strip().split('#')[-1][2:]
    if not isinstance(cap, unicode):
        enc = detect(cap)['encoding']
        if enc != 'utf-8':
            cap = cap.decode(enc).encode('utf-8')
    else:
        cap.encode('utf-8')
    filename = captions[i].strip().split('#')[0]
    id2caption[name2id[name]] = cap
    if filename not in filename2caption:
        filename2caption[filename] = [cap]
    else:
        filename2caption[filename].append(cap)
    # print filename2caption
    
cPickle.dump(filename2caption, open('filename2caption.pkl','wb'))    
cPickle.dump(id2caption, open('id2caption.pkl', 'wb'))

# generate split dictionary
train_path = './AIchallengetrain.txt'
test_path = './AIchallengetest.txt'
val_path = './AIchallengeval.txt'
splits = {}
splits['train_name'] = open(train_path).read().splitlines()
print 'num of train:', len(open(train_path).read().splitlines())
splits['test_name'] = open(test_path).read().splitlines()
splits['val_name'] = open(val_path).read().splitlines()

splits['train_id'] = [name2id[n] for n in splits['train_name']]
splits['test_id'] = [name2id[n] for n in splits['test_name']]
splits['val_id'] = [name2id[n] for n in splits['val_name']]

cPickle.dump(splits, open('splits.pkl', 'wb'))
