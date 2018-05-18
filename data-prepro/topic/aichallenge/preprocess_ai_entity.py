import re
import pickle
import json
import numpy as np
from tqdm import tqdm
import pdb
import sys

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
else:
    print 'error phase'
    print 'either train or val'
    exit

split_id = load_txt(split_path)

id2name = {}
name2id = {}
id2caption = {}
description_list = []
topic_list = []
img_name = []

# lemmatizer = WordNetLemmatizer()
data_path = './seg.AIchallenge.caption.txt'
data = load_txt(data_path)
for info in data:
    info_id = info.split('#')[0]
    if info_id in split_id:
        id2name[info_id] = info_id
        name2id[info_id] = info_id
        id2caption[info_id] = []
count = 0
for k in tqdm(range(len(data))):
    topic = []
    image_id, _, sen = data[k].split('#')
    sen = ''.join(sen[2:].split()) # improtant !
    for word, pos in thul.cut(sen):
        # print word, pos
        if pos in ['n', 'v']:
            topic.append(word)
    if image_id in split_id:
        id2caption[image_id].append(sen)
        file_name = id2name[image_id]
        description_list.append(sen)
        img_name.append(file_name)
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

