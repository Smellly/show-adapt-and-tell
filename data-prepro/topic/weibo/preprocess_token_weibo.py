# encdoe:utf-8
import re
import json
import numpy as np
from tqdm import tqdm
# import pdb
import os
try: 
    import cPickle as pkl
except:
    import pickle as pkl
import string
import sys
from chardet import detect

def unpickle(p):
    return pkl.load(open(p,'r'))

def load_json(p):
    return json.load(open(p,'r'))

def clean_words(data):
    d = {}
    freq = {}
    # start with 1
    idx = 1
    sentence_count = 0
    eliminate = 0
    # max_w = 30
    # for k in tqdm(range(len(data['caption']))):
    print 'cleaning words...'
    for k in tqdm(data.keys()):
        # print k, type(k), data[k]
        sen = data[k][0]['caption'].strip()
        if not isinstance(sen, unicode):
            enc = detect(sen)['encoding']
            if enc != 'utf-8':
                sen = sen.decode(enc).encode('utf-8')
        else:
            sen.encode('utf-8')
        filename = data[k][0]['filename']
        # skip the no image description
        words = re.split(' ', sen)
        # pop the last u'.'
        # n = len(words)
        # if n <= max_w:
        sentence_count += 1
        for word in words:
            if word not in d.keys():
                d[word] = idx
                idx += 1
                freq[word] = 1
            else:
                freq[word] += 1
        # else:
        #     eliminate += 1

    # print 'Threshold(max_words) =', max_w
    print 'Eliminate =', eliminate 
    print 'Total sentence_count =', sentence_count
    print 'Number of different words =', len(d.keys())
    print 'Saving....'

    np.savez('K_cleaned_words', dict=d, freq=freq)

    return d, freq

phase = sys.argv[1]
print 'phase : ', phase
data_path = './K_' + phase + '_annotation.json'
# data = unpickle(data_path)
data = load_json(data_path)
# print type(data), data.keys()

id2name = unpickle('id2name.pkl')
id2caption = unpickle('id2caption.pkl')
id2topic = unpickle('id2topic.pkl')
splits = unpickle('splits.pkl')
split = splits[phase + '_id']
thres = 1

filename_list = []
caption_list = []
topic_list = []
img_id_list = []
for i in split:
    # print i
    sen = id2caption[i]
    # print sen
    if not isinstance(sen, unicode):
        enc = detect(sen)['encoding']
        if enc != 'utf-8':
            sen = sen.decode(enc).encode('utf-8')
    else:
        sen.encode('utf-8')
    img_id_list.append(i)
    filename_list.append(id2name[i])
    caption_list.append(sen)
    for tpc in id2topic[i]:
        topic_list.append(tpc)
    # print caption_list
    # print topic_list
    # break

print 'num of topic_list from splits:', len(topic_list)
print 'num of caption_list from splits:', len(caption_list)

# build dictionary
if not os.path.isfile('./ai&weibo_dictionary_'+str(thres)+'.npz'):
    # pdb.set_trace()
    # clean the words through the frequency
    if not os.path.isfile('K_cleaned_words.npz'):
	d, freq = clean_words(data)
    else:
        words = np.load('K_cleaned_words.npz')
        d = words['dict'].item(0)
        freq = words['freq'].item(0)

    print 'CUB words in the K_clean_words =', len(d.keys())
    
    tem = np.load('../aichallenge/dictionary_5.npz')
    word2idx = tem['word2idx'].item(0)
    idx2word = tem['idx2word'].item(0)
    tmp = len(word2idx.keys())
    print 'MSCOCO words in the dictionary =', len(word2idx.keys())

    idx = len(word2idx)
    underThresh = 0
    for k in tqdm(d.keys()):
        if freq[k] >= thres and k not in word2idx:
            word2idx[k] = idx
            idx2word[str(idx)] = k
            idx += 1
        elif k not in word2idx:
            underThresh += 1

    print 'Threshold of word fequency =', thres
    print 'CUB words in the dictionary =', len(word2idx.keys()) - tmp
    np.savez('./ai&weibo_dictionary_'+str(thres), word2idx=word2idx, idx2word=idx2word)

else:
    tem = np.load('./ai&weibo_dictionary_'+str(thres)+'.npz')
    word2idx = tem['word2idx'].item(0)
    idx2word = tem['idx2word'].item(0)

print 'Total words in the dictionary =', len(word2idx.keys())

# generate tokenized data
num_sentence = 0
num_topic = 0
eliminate = 0
tokenized_caption_list = []
tokenized_topic_list = []
caption_list_new = []
topic_list_new = []
filename_list_new = []
img_id_list_new = []
# caption_length = []
# topic_length = []

for k in tqdm(xrange(len(caption_list))):
    # print k, caption_list[k].strip()
    sen = caption_list[k].strip()
    if not isinstance(sen, unicode):
        enc = detect(sen)['encoding']
        if enc != 'utf-8':
            sen = sen.decode(enc).encode('utf-8')
    else:
        sen.encode('utf-8')
    img_id = img_id_list[k]
    filename = filename_list[k]
    topics = topic_list[k]
    words = re.split(' ', sen)
    # print words, topics
    count = 0
    tokenized_sent = np.ones([31],dtype=int) * word2idx[u'<NOT>']  # initialize as <NOT>
    tokenized_topic = np.zeros([7], dtype=int) # max topix word is 7
    newtopic = []
    valid = False
    if len(words) <= 30:
        valid = True
        for word in words:
            try:
                idx = int(word2idx[word])
                tokenized_sent[count] = idx
                count += 1
            except KeyError:
                # if contain <UNK> then drop the sentence in train phase
                print k, sen
                print word
                valid = False
                break
        # add <EOS>
        tokenized_sent[len(words)] = word2idx[u'<EOS>']
        if valid:
            tokenized_caption_list.append(tokenized_sent)
            filename_list_new.append(filename)
            img_id_list_new.append(img_id)
            caption_list_new.append(sen)
            num_sentence += 1
        else:
            eliminate += 1  
    if valid:
        for ind, topic in enumerate(topics):
            if ind == 7:
                break
            try:
                idx = int(word2idx[topic])
                tokenized_topic[ind] = idx
                newtopic.append(topic)
            except KeyError:
                pass
        if valid:
            # tokenized_topic[count] = (word2idx["<EOS>"])
            topic_list.append(newtopic)
            # length = np.sum((tokenized_sent!=0)+0)
            tokenized_topic_list.append(tokenized_topic)
            # topic_length.append(length)
            num_topic += 1

print 'Number of sentence =', num_sentence
print 'Number of topic =', num_topic
print 'eliminate = ', eliminate

assert num_topic==num_sentence

tokenized_caption_info = {}
tokenized_caption_info['tokenized_caption_list'] = np.asarray(tokenized_caption_list)
tokenized_caption_info['tokenized_topic_list'] = np.asarray(tokenized_topic_list)
tokenized_caption_info['filename_list'] = np.asarray(filename_list_new)
tokenized_caption_info['img_id_list'] = np.asarray(img_id_list_new)
tokenized_caption_info['raw_caption_list'] = np.asarray(caption_list_new)
tokenized_caption_info['raw_topic_list'] = np.asarray(topic_list_new)


with open('./tokenized_'+phase+'_caption.pkl', 'w') as outfile:
    pkl.dump(tokenized_caption_info, outfile)

