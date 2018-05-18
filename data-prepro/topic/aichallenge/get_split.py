import numpy as np
import os
import cPickle
from tqdm import tqdm

# generate name2id & id2name dictionary
name_id_path = './AIchallengeSet.txt'
name_id = open(name_id_path).read().splitlines()
name2id = {}
id2name = {}
for i in range(len(name_id)):
    name2id[name_id[i]] = str(i)
    id2name[str(i)] = name_id[i]

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
files = set()
for i in tqdm(range(len(captions))):
    #print captions[i].strip().split('####')[1]
    cap = captions[i].strip().split('####')[1]     
    filename = captions[i].strip().split('####')[0]
    id2caption[str(i)] = cap
    if filename not in files:
        files.add(filename)
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
print len(open(train_path).read().splitlines())
splits['test_name'] = open(test_path).read().splitlines()
splits['val_name'] = open(val_path).read().splitlines()

splits['train_id'] = [name2id[n] for n in splits['train_name']]
splits['test_id'] = [name2id[n] for n in splits['test_name']]
splits['val_id'] = [name2id[n] for n in splits['val_name']]

cPickle.dump(splits, open('splits.pkl', 'wb'))

