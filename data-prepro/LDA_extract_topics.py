# -*- coding: utf-8 -*-
from __future__ import print_function
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

import json

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

def getCaption(json_file):
    content = []
    for img_id in json_file:
        # print 'img_id:', img_id
        for cap in json_file[img_id]:
            content.append(cap['caption'])
        # break
    return content

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

def LDA(content):
    # vectorize
    n_features = 1000
    tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words='english',
                                min_df = 1)
    tf = tf_vectorizer.fit_transform(content)

    for n_topics in range(10, 20):
        lda = LatentDirichletAllocation(
                                n_components=n_topics, 
                                max_iter=500,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
        lda.fit(tf)
        n_top_words = 20
        tf_feature_names = tf_vectorizer.get_feature_names()
        print_top_words(lda, tf_feature_names, n_top_words)
        

if __name__ == '__main__':
    cub_path = '/home/smelly/projects/show-adapt-and-tell/data-prepro/CUB200_preprocess/cub_data/K_train_annotation.json'
    # captions_json = ReadJson(cub_path)
    # captions = getCaption(captions_json)
    # LDA(captions)
    mscoco_path = '/home/smelly/projects/show-adapt-and-tell/data-prepro/MSCOCO_preprocess/mscoco_data/K_train_annotation.json'
    captions_json = ReadJson(mscoco_path)
    captions = getCaption(captions_json)
    LDA(captions)
        
