# encoding:utf-8
import re
import pickle
import json
import numpy as np
from tqdm import tqdm
import sys
# from chardet import detect
from enuncoding import *

# Chinense
import thulac
thul = thulac.thulac()

def load_txt(path):
    with open(path, 'r') as f:
        raw = f.readlines()
    r = [x.strip() for x in raw]
    return r

desired_phase = sys.argv[1]
if desired_phase == 'train':
    split_path = './AIchallengetrain.txt'
elif desired_phase == 'val':
    split_path = './AIchallengeval.txt'
elif desired_phase == 'test':
    split_path = './AIchallengetest.txt'
else:
    print 'error phase'
    print 'either train or val'
    exit

split_name_list = load_txt(split_path)
split_name = {x:1 for x in split_name_list}

id2name = pickle.load(open('./id2name.pkl'))
name2id = pickle.load(open('./name2id.pkl'))
id2caption = pickle.load(open('./id2caption.pkl'))
description_list = []
topic_list = []
img_name = []

data_path = './seg.AIchallenge.caption.txt'
data = load_txt(data_path)
'''
for info in tqdm(data):
    info_id = info.split('#')[0]
    if info_id in split_name:
        id2name[name2id_all[info_id]] = info_id
        name2id[info_id] = name2id_all[info_id]
        id2caption[info_id] = []
'''

count = 0
print 'num of split_name %s : %d'%(desired_phase, len(split_name))
for k in tqdm(xrange(len(data))):
    file_name, _, sen = data[k].split('#')
    if file_name in split_name:
        image_id = name2id[file_name]
        # id2caption[image_id].append(sen)
        description_list.append(sen)
        img_name.append(file_name)
        topic = []
        sen_t = encode_utf8(sen)
        for word, pos in thul.cut(sen_t):
            if pos in ['n', 'v']:
                topic.append(decode_any(word))
        topic_list.append(topic)
    # break

out = {}
out['caption_entity'] = description_list
out['file_name'] = img_name
out['id2filename'] = id2name
out['filename2id'] = name2id
out['id2caption'] = id2caption
out['topic_entity'] = topic_list

print 'Saving ...'
print 'Numer of sentence =', len(description_list)      
with open('./K_annotation_%s.pkl'%desired_phase, 'w') as outfile:
    pickle.dump(out, outfile)

