# encoding:utf-8
import numpy as np
import cPickle
from tqdm import tqdm
from chardet import detect
import sys
sys.path.append('../aichallenge')
default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)
from enuncoding import *

# Chinese
import thulac
thul = thulac.thulac()

# generate name2id & id2name dictionary
id2caption = {}
id2topic = {}
name2id = {}
id2name = {}
splits = {}
splits['train_name'] = []
splits['val_name'] = []
splits['test_name'] = []
splits['train_id'] = []
splits['val_id'] = []
splits['test_id'] = []
count = 0

for phase in ['train', 'val', 'test']:
    print 'phase : %s'%phase
    name_id_path = './seg.weiboclear'+phase+'V4.caption.txt'
    name_id = open(name_id_path).read().splitlines()
    for item in tqdm(name_id):
        try:
            name, sen = item.split('####')
        except:
            print item
            continue
        sen = encode_utf8(sen)
        name2id[name] = count
        id2name[count] = name
        id2caption[count] = sen
        id2topic[count] = []
        topic = []
        for word, pos in thul.cut(''.join(sen.split())):
            if pos in ['n', 'v']:
                # print word, pos
                try:
                    topic.append(decode_any(word))
                except:
                    pass
        id2topic[count].append(topic)
        if phase == 'train':
            splits['train_name'].append(name)
            splits['train_id'].append(count)
        elif phase == 'val':
            splits['val_name'].append(name)
            splits['val_id'].append(count)
        elif phase == 'test':
            splits['test_name'].append(name)
            splits['test_id'].append(count)
        count += 1

print 'len of id2topic:', len(id2topic)
print 'len of id2caption:', len(id2caption)
assert len(id2topic) == len(id2caption)
cPickle.dump(name2id, open('name2id.pkl', 'wb'))
cPickle.dump(id2name, open('id2name.pkl', 'wb'))
cPickle.dump(id2caption, open('id2caption.pkl', 'wb'))
cPickle.dump(id2topic, open('id2topic.pkl', 'wb'))
cPickle.dump(splits, open('splits.pkl', 'wb'))

