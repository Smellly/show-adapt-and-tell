# encdoing: utf-8
import re
# import json
import numpy as np
from tqdm import tqdm
import pickle
import cPickle
import sys
from chardet import detect

def unpickle(p):
    return cPickle.load(open(p,'r'))

'''
def load_json(p):
    return json.load(open(p,'r'))
'''

def clean_words(data):
    print 'generating clean words...'
    dict = {}
    freq = {}
    # start with 1
    idx = 1
    sentence_count = 0
    eliminate = 0
    max_w = 30
    for k in tqdm(xrange(len(data['caption_entity']))):
        sen = data['caption_entity'][k]
        if not isinstance(sen, unicode):
            enc = detect(sen)['encoding']
            if enc != 'utf-8':
                sen = sen.decode(enc).encode('utf-8')
        else:
            sen.encode('utf-8')
        filename = data['file_name'][k]
        # skip the no image description
        words = re.split(' ', sen)
        # pop the last u'.'
        n = len(words)
        if "" in words:
            words.remove("")
        if n <= max_w:
            sentence_count += 1
            for word in words:
                if "\n" in word:
                    word = word.replace("\n", "")
                if word not in dict.keys():
                    dict[word] = idx
                    idx += 1
                    freq[word] = 1
                else:
                    freq[word] += 1
        else:
            eliminate += 1

    print 'Threshold(max_words) =', max_w
    print 'Eliminate =', eliminate 
    print 'Total sentence_count =', sentence_count
    print 'Number of different words =', len(dict.keys())
    print 'Saving....'

    np.savez('K_cleaned_words', dict=dict, freq=freq)
    return dict, freq

phase = sys.argv[1]
data_path = './K_annotation_'+phase+'.pkl'
data = unpickle(data_path)
thres = 5

# create word2idx and idx2word
# we should use the whole mscoco word here
if not os.path.isfile('./dictionary_'+str(thres)+'.npz'):
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
        if freq[k] >= thres and k != "":
            word2idx[k] = idx
            idx2word[str(idx)] = k
            idx += 1

    word2idx[u'<BOS>'] = 0
    idx2word["0"] = u'<BOS>'
    word2idx[u'<EOS>'] = len(word2idx.keys())
    idx2word[str(len(idx2word.keys()))] = u'<EOS>'
    word2idx[u'<UNK>'] = len(word2idx.keys())
    idx2word[str(len(idx2word.keys()))] = u'<UNK>'
    word2idx[u'<NOT>'] = len(word2idx.keys())
    idx2word[str(len(idx2word.keys()))] = u'<NOT>'

    print 'Threshold of word fequency =', thres
    print 'Total words in the dictionary =', len(word2idx.keys())
    np.savez('./dictionary_'+str(thres), word2idx=word2idx, idx2word=idx2word)
else:
    tem = np.load('./dictionary_'+str(thres)+'.npz')
    word2idx = tem['word2idx'].item(0)
    idx2word = tem['idx2word'].item(0)

num_sentence = 0
num_topic = 0
eliminate = 0
tokenized_caption_list = []
tokenized_topic_list = []
caption_list = []
topic_list = []
filename_list = []
caption_length = []
topic_length = []

print 'processing...'
for k in tqdm(xrange(len(data['caption_entity']))):
    sen = data['caption_entity'][k]
    if not isinstance(sen, unicode):
        enc = detect(sen)['encoding']
        if enc != 'utf-8':
            sen = sen.decode(enc).encode('utf-8')
    else:
        sen.encode('utf-8')
    filename = data['file_name'][k]
    topics = data['topic_entity'][k]
    # skip the no image description
    words = re.split(' ', sen)
    # pop the last u'.'
    tokenized_sent = np.zeros([30+1], dtype=int) # max sent words is 30
    tokenized_sent.fill(int(word2idx[u'<NOT>']))
    tokenized_topic = np.zeros([7], dtype=int) # max topic words is 7
    #tokenized_sent[0] = int(word2idx[u'<BOS>'])
    count = 0
    # topic_count = 0
    caption = []
    newtopic = []
    valid = False

    if len(words) <= 30:
        valid = True
        for word in words:
            try:
                if word != "":
                    idx = int(word2idx[word])
                    tokenized_sent[count] = idx
                    caption.append(word)
                    count += 1
            except KeyError:
                # if contain <UNK> then drop the sentence
                if phase == 'train':
                    valid = False
                    break
                else:
                    tokenized_sent[count] = int(word2idx[u'<UNK>'])
                    count += 1
        if valid:
            tokenized_sent[count] = (word2idx["<EOS>"]) # the end of a sentence
            caption_list.append(caption)
            length = np.sum((tokenized_sent!=0)+0)
            tokenized_caption_list.append(tokenized_sent)
            filename_list.append(filename)
            caption_length.append(length)
            num_sentence += 1
        else:
            # if phase == 'val':
            #         pdb.set_trace()
            eliminate += 1  

    if valid: # and len(topics) < 7+1:
        for ind, topic in enumerate(topics):
            if ind == 7:
                break
            try:
                idx = int(word2idx[topic])
                tokenized_topic[ind] = idx
                newtopic.append(topic)
                # topic_count += 1
            except KeyError:
                pass
        if valid:
            topic_list.append(newtopic)
            length = np.sum((tokenized_sent!=0)+0)
            tokenized_topic_list.append(tokenized_topic)
            topic_length.append(length)
            num_topic += 1

print "num topic:", num_topic
print "num sentence:", num_sentence
assert num_topic == num_sentence

tokenized_caption_info = {}
tokenized_caption_info['caption_length'] = np.asarray(caption_length)
tokenized_caption_info['topic_length'] = np.asarray(topic_length)
tokenized_caption_info['tokenized_caption_list'] = np.asarray(tokenized_caption_list)
tokenized_caption_info['tokenized_topic_list'] = np.asarray(tokenized_topic_list)
tokenized_caption_info['caption_list'] = np.asarray(caption_list)
tokenized_caption_info['topic_list'] = np.asarray(topic_list)
tokenized_caption_info['filename_list'] = np.asarray(filename_list)

print 'Number of sentence =', num_sentence
with open('./tokenized_'+phase+'_caption.pkl', 'w') as outfile:
        pickle.dump(tokenized_caption_info, outfile)

