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
    max_w = 30
    # for k in tqdm(range(len(data['caption']))):
    for k in tqdm(data.keys()):
        # print k, type(k), data[k]
        sen = data[k][0]['caption']
        filename = data[k][0]['filename']
        # skip the no image description
        words = re.split(' ', sen)
        # pop the last u'.'
        n = len(words)
        if n <= max_w:
            sentence_count += 1
            for word in words:
                for p in string.punctuation:
                    if p in word:
                        word = word.replace(p,'')
                word = word.lower()
                if word not in d.keys():
                    d[word] = idx
                    idx += 1
                    freq[word] = 1
                else:
                    freq[word] += 1
        else:
            eliminate += 1

    print 'Threshold(max_words) =', max_w
    print 'Eliminate =', eliminate 
    print 'Total sentence_count =', sentence_count
    print 'Number of different words =', len(d.keys())
    print 'Saving....'

    np.savez('K_cleaned_words', dict=d, freq=freq)

    return d, freq

phase = sys.argv[1]
print 'phase : ', phase
data_path = '../cub/K_' + phase + '_annotation.json'
# data = unpickle(data_path)
data = load_json(data_path)
# print type(data), data.keys()

id2name = unpickle('id2name.pkl')
id2caption = unpickle('id2caption.pkl')
id2topic = unpickle('id2topic.pkl')
splits = unpickle('splits.pkl')
split = splits[phase + '_id']
thres = 5

filename_list = []
caption_list = []
topic_list = []
topic_list = []
img_id_list = []
for i in split:
    for sen in id2caption[i]:
        img_id_list.append(i)
        filename_list.append(id2name[i])
        caption_list.append(sen)
    for tpc in id2topic[i]:
        topic_list.append(tpc)

# print topic_list

# build dictionary
if not os.path.isfile('../cub/dictionary_'+str(thres)+'.npz'):
    # pdb.set_trace()
    # clean the words through the frequency
    if not os.path.isfile('K_cleaned_words.npz'):
	d, freq = clean_words(data)
    else:
        words = np.load('K_cleaned_words.npz')
        d = words['dict'].item(0)
        freq = words['freq'].item(0)

    idx2word = {}
    word2idx = {}
    idx = 1
    for k in tqdm(d.keys()):
        if freq[k] >= thres:
            word2idx[k] = idx
            idx2word[str(idx)] = k
            idx += 1

    word2idx[u'<BOS>'] = 0
    idx2word["0"] = u'<BOS>'
    word2idx[u'<EOS>'] = len(word2idx.keys())
    idx2word[str(len(idx2word.keys()))] = u'<EOS>'
    word2idx[u'<UNK>'] = len(word2idx.keys())
    idx2word[str(len(word2idx.keys()))] = u'<UNK>'
    word2idx[u'<NOT>'] = len(word2idx.keys())
    idx2word[str(len(idx2word.keys()))] = u'<NOT>'

    print 'Threshold of word fequency =', thres
    print 'Total words in the dictionary =', len(word2idx.keys())
    np.savez('../cub/dictionary_'+str(thres), word2idx=word2idx, idx2word=idx2word)
else:
    tem = np.load('../cub/dictionary_'+str(thres)+'.npz')
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

for k in tqdm(range(len(caption_list))):
    sen = caption_list[k]
    img_id = img_id_list[k]
    filename = filename_list[k]
    topics = topic_list[k]
    # skip the no image description
    words = re.split(' ', sen)
    # pop the last u'.'
    count = 0
    tokenized_sent = np.ones([31],dtype=int) * word2idx[u'<NOT>']  # initialize as <NOT>
    tokenized_topic = np.zeros([7], dtype=int) # max topix word is 7
    newtopic = []
    valid = False
    if len(words) <= 30:
        valid = True
        for word in words:
            try:
                word = word.lower()
                for p in string.punctuation:
                    if p in word:
                        word = word.replace(p,'')
                idx = int(word2idx[word])
                tokenized_sent[count] = idx
                count += 1
            except KeyError:
                # if contain <UNK> then drop the sentence in train phase
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
                topic = topic.lower()
                for p in string.punctuation:
                    if p in topic:
                        topic = topic.replace(p,'')
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


with open('../cub/tokenized_'+phase+'_caption.pkl', 'w') as outfile:
    pkl.dump(tokenized_caption_info, outfile)

