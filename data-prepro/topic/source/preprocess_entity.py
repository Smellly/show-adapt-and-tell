import re
import pickle
import json
import numpy as np
from tqdm import tqdm
import pdb
import sys

# from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
# from nltk.stem import WordNetLemmatizer

def load_json(p):
    return json.load(open(p,'r'))

'''
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
'''

person = [
        'man', 'men', 'people', 'person', 'woman', 'women', 'child', 'children',
        'baby', 'guy', 'husband', 'wife', 'brother', 'sister', 'gentleman', 
        'policeman', 'fireman', 'bussinessman', 'human', 'fisherman', 'cameraman',
        'batman', 'serviceman', 'workman', 'salesman', 'boy', 'girl'
        ]

desired_phase = sys.argv[1]
split_path = 'K_split.json'
split = load_json(split_path)
split_id = split[desired_phase]

phase = ['train', 'val']
id2name = {}
name2id = {}
id2caption = {}
description_list = []
topic_list = []
img_name = []

for p in phase:
    # lemmatizer = WordNetLemmatizer()
    data_path = '../mscoco_person_data/captions_person_%s2014.json' % p
    data = load_json(data_path)
    for img_info in data['images']:
        if img_info['id'] in split_id:
            id2name[str(img_info['id'])] = img_info['file_name']
            name2id[img_info['file_name']] = str(img_info['id'])
            id2caption[str(img_info['id'])] = []
    count = 0
    for k in tqdm(range(len(data['annotations']))):
        tag = False
        topic = []
        sen = data['annotations'][k]['caption']
        for word, pos in pos_tag(word_tokenize(sen)):
            if word in person:
                tag = True
            if pos.startswith('N') or pos.startswith('V'):
                # word_lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos))
                topic.append(word)
        image_id = data['annotations'][k]['image_id']
        if tag and image_id in split_id:
            id2caption[str(image_id)].append(sen)
            file_name = id2name[str(image_id)]
            description_list.append(sen)
            img_name.append(file_name)
            topic_list.append(topic)

out = {}
out['caption_entity'] = description_list
out['file_name'] = img_name
out['id2filename'] = id2name
out['filename2id'] = name2id
out['id2caption'] = id2caption
out['topic_entity'] = topic_list

print 'Saving ...'
print 'Numer of sentence =', len(description_list)      
with open('../mscoco_person_data/K_annotation_%s2014.pkl'%desired_phase, 'w') as outfile:
    pickle.dump(out, outfile)

