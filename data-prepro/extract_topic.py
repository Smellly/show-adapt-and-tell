# -*- coding: utf-8 -*-
'''
// Author: Jay Smelly.
// Last modify: 2018-04-24 16:15:52.
// File name: extract_topic.py
//
// Description:
'''
"""
Created on 2016.08.27
@autho Jay Smelly
"""
from sklearn.decomposition import LatentDirichletAllocation

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

from tqdm import tqdm

import json
import cPickle as pkl

def Read(path):
    adjs = []
    with open(path, 'r') as fr:
        for raw in fr:
            adjs.append(raw.replace(' \r\n',''))
    return adjs

def ReadJson(path):
    with open(path, 'r') as f:
        j = json.load(f)
    return j

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

def genWordDict(json_file):
    wordDict = dict()
    lemmatizer = WordNetLemmatizer()

    person = [
            'man', 'men', 'people', 'person', 'woman', 'women', 'child', 'children',
            'baby', 'guy', 'husband', 'wife', 'brother', 'sister', 'gentleman', 
            'policeman', 'fireman', 'bussinessman', 'human', 'fisherman', 'cameraman',
            'batman', 'serviceman', 'workman', 'salesman', 'boy', 'girl'
            ]

    sports = [
            'tennis', 'baseball', 'ball', 'player', 'court', 'game', 'bat', 'lot',
            'racket', 'sport', 'batman', 'baseman']

    new_json = dict()

    tag = False
    for img_id in tqdm(json_file):
        # print 'img_id:', img_id
        for cap in json_file[img_id]:
            # print 'cap:', cap
            for word, pos in pos_tag(word_tokenize(cap['caption'])):
                # print 'word, pos:', word, pos
                if word in person:
                    tag = True
                if img_id not in new_json:
                    new_json[img_id] = []
                if 'themes' not in cap:
                    # print 'themes init'
                    cap['themes'] = []
                if pos.startswith('N') or pos.startswith('V'):
                    word_lemma = lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos))
                    # print 'word_lemma:', word_lemma
                    cap['themes'].append(word_lemma)
                    # if word_lemma not in wordDict:
                    #     wordDict[word_lemma] = 1
                    # else:
                    #     wordDict[word_lemma] += 1
            # print 'cap:', cap
            if tag:
                new_json[img_id].append(cap)
                tag = False
        # break
    # print sorted(wordDict.items(), key=lambda x:x[1])
    return new_json

def FindSynonyms(adjs, path):
    synonym_list = dict()
    for word in adjs:
        synonyms = set()
        # print 'word:',word
        for syn in wordnet.synsets(word):
            # print 'syn:',syn
            for l in syn.lemmas():
                synonyms.add(l.name().encode('utf-8'))
                # print 'l.name():',l.name()
        if synonyms:
            synonym_list[word] = list(synonyms)
        
    with open(path, 'w') as fw:
        pre = synonym_list.items() # pre is a list
        for ind,content in pre:
            # print ind,content
            content = str(content)
            content = content.replace('[','').replace(']','\n').replace("'",'')
            fw.write(ind)
            fw.write('#')
            fw.write(content)

    return synonym_list

if __name__ == '__main__':
        cub_path = '/home/smelly/projects/show-adapt-and-tell/data-prepro/CUB200_preprocess/cub_data/K_train_annotation.json'
        mscoco_path = '/home/smelly/projects/show-adapt-and-tell/data-prepro/MSCOCO_preprocess/mscoco_data/K_train_annotation.json'
        captions_json = ReadJson(mscoco_path)
        # captions_json = ReadJson(cub_path)
        new_json = genWordDict(captions_json)
        with open('mscoco_person.json', 'w') as f:
            json.dump(new_json, f)
        # with open('mscoco_person_words.pkl', 'w') as f:
        #     pkl.dump(words, f)
        # with open('mscoco_person_wordList.txt', 'w') as f:
        #     for word in sorted(words.items(), key=lambda x:x[1], reverse=True):
        #         try:
        #             f.write('%s#%d\n'%(word[0], word[1]))
        #         except:
        #             pass
        # antonyms = FindSynonyms(words, 'words_antonyms.txt')
        # print 'antonyms:',antonyms
