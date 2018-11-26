# encoding: utf-8
import numpy as np
import utils
import os, re, json
# import pdb
from tqdm import tqdm
import sys
sys.path.append('./coco_spice/pycocotools/')
import coco

def get_key(name):
    return re.split('\.', name)[0]

class mscoco_negative():
    def __init__(self, dataset, conf):
        self.dataset_name = 'mscoco_negative'
        self.batch_size = conf.batch_size
        data_dir = './negative_samples/mscoco_sample'
        npz_paths = os.listdir(data_dir)
        print "Load Training data"
        count = 0
        self.neg_img_filename_train = []
        for npz_path in tqdm(npz_paths):
            if int(re.split("\.", re.split("_", npz_path)[1])[0]) <= 30000:
                npz = np.load(os.path.join(data_dir, npz_path))
                # tokenize caption
                if count == 0:
                    self.neg_caption_train = npz["index"]
                else:
                    self.neg_caption_train = np.concatenate((self.neg_caption_train, npz["index"]), 0)
                # img_idx
                for i in npz["img_name"]:
                    self.neg_img_filename_train.append(i+'.jpg')
                count += 1
        self.neg_img_filename_train = np.asarray(self.neg_img_filename_train)

        npz_paths = ["mscoco_51000.npz"]
        print "Testing data"
        self.neg_img_filename_test = []
        count = 0
        for npz_path in tqdm(npz_paths):
            npz = np.load(os.path.join(data_dir, npz_path))
            if count == 0:
                self.neg_caption_test = npz["index"]
            else:
                self.neg_caption_test = np.concatenate((self.neg_caption_test, npz["index"]), 0)
            # img_idx
            for i in npz["img_name"]:
                self.neg_img_filename_test.append(i+'.jpg')
            count += 1
        self.neg_img_filename_test = np.asarray(self.neg_img_filename_test)

        self.current = 0
        self.num_train = len(self.neg_img_filename_train)
        self.num_test = len(self.neg_img_filename_test)
        self.random_shuffle()
        self.filename2id = dataset.filename2id
        self.img_dims = dataset.img_dims
        self.img_feat = dataset.img_feat

    def random_shuffle(self):
        idx = range(self.num_train)
        np.random.shuffle(idx)
        self.neg_img_filename_train = self.neg_img_filename_train[idx]
        self.neg_caption_train = self.neg_caption_train[idx, :]

    def get_paired_data(self, num_data, phase):
        if phase == 'train':
            caption = self.neg_caption_train
            img_filename = self.neg_img_filename_train
        else:
            caption = self.neg_caption_test
            img_filename = self.neg_img_filename_test

        if num_data > 0:
            caption = caption[:num_data, :]
            img_filename = img_filename[:num_data]
        else:
            if phase=='train':
                num_data = self.num_train
            else:
                num_data = self.num_test

        image_feature = np.zeros([num_data, self.img_dims])
        img_idx = []
        for i in range(num_data):
            # image_feature[i, :] = self.img_feat[get_key(img_filename[i])]
            image_feature[i, :] = self.img_feat[i]
            img_idx.append(get_key(img_filename[i]))
        return image_feature, caption, np.asarray(img_idx)

    def sequential_sample(self, batch_size):
        end = (self.current+batch_size) % self.num_train
        if self.current + batch_size < self.num_train:
            caption = self.neg_caption_train[self.current:end, :]
            img_filename = self.neg_img_filename_train[self.current:end]
        else:
            caption = np.concatenate((self.neg_caption_train[self.current:], self.neg_caption_train[:end]), axis=0)
            img_filename = np.concatenate((self.neg_img_filename_train[self.current:], self.neg_img_filename_train[:end]), axis=0)
            self.random_shuffle()

        image_feature = np.zeros([batch_size, self.img_dims])
        img_id = []
        for i in range(batch_size):
            # image_feature[i, :] = self.img_feat[get_key(img_filename[i])]
            image_feature[i, :] = self.img_feat[i]
            img_id.append(self.filename2id[img_filename[i]])
        self.current = end
        return image_feature, caption, np.asarray(img_id)

class mscoco():
    def __init__(self, conf=None):
        # train img feature
        self.dataset_name = 'cub'
        self.GPretrainedDatasetName = 'mscoco'
        # target data
        # flickr_img_path = './cub/cub_trainval_feat.pkl' # todo topic
        # self.num_train_images_filckr = len(self.train_flickr_img_feat.keys())
        # self.t rain_img_idx = self.train_flickr_img_feat.keys()
        flickr_caption_train_data_path = './cub/tokenized_train_caption.pkl'
        flickr_caption_train_data = utils.unpickle(flickr_caption_train_data_path)
        self.flickr_caption_train = flickr_caption_train_data['tokenized_caption_list']
        self.train_flickr_img_feat = flickr_caption_train_data['tokenized_topic_list']
        self.train_flickr_img_id = flickr_caption_train_data['img_id_list']
        self.num_train_images_flickr = len(self.train_flickr_img_feat)
        self.train_img_idx = range(self.num_train_images_flickr)
        self.flickr_caption_idx_train = flickr_caption_train_data['filename_list']
        self.num_flickr_train_caption = self.flickr_caption_train.shape[0]
        # flickr_testimg_path = './cub/cub_test_feat.pkl'
        flickr_testimg_path = './cub/tokenized_test_caption.pkl'
        self.test_flickr_img_feat = utils.unpickle(flickr_testimg_path)['tokenized_topic_list']
        self.flickr_random_shuffle()    # shuffle the text data
        self.flickr_id2topic = utils.unpickle('./cub/id2topic.pkl')

        # MSCOCO data
        # img_feat_path = './data/coco_trainval_feat.pkl' # todo topic
        # self.img_feat = utils.unpickle(img_feat_path)
        train_meta_path = './data/K_annotation_train2014.pkl'
        # train_meta_path = './data/K_train_annotation.pkl'
        train_meta = utils.unpickle(train_meta_path)
        self.filename2id = train_meta['filename2id']
        val_meta_path = './data/K_annotation_val2014.pkl'
        # val_meta_path = './data/K_val_annotation.pkl'
        val_meta = utils.unpickle(val_meta_path)
        self.id2filename = val_meta['id2filename']
        # train caption
        caption_train_data_path = './data/tokenized_train_caption.pkl'
        caption_train_data = utils.unpickle(caption_train_data_path)
        self.caption_train = caption_train_data['tokenized_caption_list']
        self.caption_idx_train = caption_train_data['filename_list']

        # topic data
        self.train_img_feat = caption_train_data['tokenized_topic_list']

        # val caption
        caption_test_data_path = './data/tokenized_val_caption.pkl'
        caption_test_data = utils.unpickle(caption_test_data_path)
        self.caption_test = caption_test_data['tokenized_caption_list']
        self.caption_idx_test = caption_test_data['filename_list']
        dict_path = './cub/mscoco&cub_dictionary_5.npz'
        temp = np.load(dict_path)
        self.ix2word = temp['idx2word'].item()
        self.word2ix = temp['word2idx'].item()
        # add <END> token
        if conf != None:
            self.batch_size = conf.batch_size
        self.dict_size = len(self.ix2word.keys())
        self.test_pointer = 0
        self.current_flickr = 0
        self.current_flickr_caption = 0
        self.current = 0
        self.max_words = self.caption_train.shape[1]
        self.max_themes = 7 # hardcode
        # tmp = self.img_feat[self.img_feat.keys()[0]]
        # self.img_dims = tmp.shape[0] # 2048
        self.img_dims = conf.D_hidden_size
        self.num_train = self.caption_train.shape[0]
        self.num_test = self.caption_test.shape[0]
        # Load annotation
        self.source_test_annotation = json.load(open('./data/K_val_annotation.json'))
        # self.source_test_images = self.source_test_annotation.keys()
        tmp = utils.unpickle('./data/K_annotation_val2014.pkl')
        # self.source_test_annotation = tmp['caption_entity']
        self.source_test_images = tmp['topic_entity']
        self.source_test_image_filename = tmp['file_name']
        self.source_test_filename2id = tmp['filename2id']
        # a image with 5 caption and 5 themes
        self.source_num_test_images = len(self.source_test_images)

        self.target_test_annotation = json.load(open('./cub/K_test_annotation.json'))
        # self.test_images = self.target_test_annotation.keys()
        tmp = utils.unpickle('./cub/tokenized_test_caption.pkl')
        self.target_test_images = tmp['tokenized_topic_list'] # np array
        self.target_test_image_id = tmp['img_id_list'] # np array
        self.num_target_test_images = len(self.target_test_images) # 
        self.random_shuffle()
        
    def random_shuffle(self):
        idx = range(self.num_train)
        np.random.shuffle(idx)
        self.caption_train = self.caption_train[idx]
        self.train_img_feat = self.train_img_feat[idx]
        self.caption_idx_train = self.caption_idx_train[idx]

    def flickr_random_shuffle(self):
        idx = range(self.num_flickr_train_caption)
        np.random.shuffle(idx)
        # caption_train is tokenized caption which type is numpy
        self.flickr_caption_train = self.flickr_caption_train[idx] 
        self.train_flickr_img_feat = self.train_flickr_img_feat[idx]
        self.train_flickr_img_id = self.train_flickr_img_id[idx]
        self.flickr_caption_idx_train = self.flickr_caption_idx_train[idx]

    def get_train_annotation(self):
        return self.train_annotation

    # mscoco
    def get_train_for_eval(self, num):
        image_feature = np.zeros([num, self.max_themes])
        filenames = []
        self.random_shuffle()
        for i in range(num):
            filename = get_key(self.caption_idx_train[i])
            filenames.append(filename)
            image_feature[i, :] = self.train_img_feat[i]

        return image_feature, np.asarray(filenames)

    # cub
    def get_test_for_eval(self):
        image_feature = np.zeros([self.num_target_test_images, self.max_themes])
        image_id = np.zeros([self.num_target_test_images], dtype=int)
        # for i in range(self.num_target_test_images):
        #     image_feature[i, :] = self.test_flickr_img_feat[self.target_test_images[i]]
        #     image_id[i] = int(self.target_test_images[i])
        for ind, i in enumerate(self.target_test_images):
            image_feature[ind, :] = i
            image_id[ind] = int(self.target_test_image_id[ind])
        return image_feature, image_id, self.target_test_annotation

    # cub
    def get_specific_for_eval(self, topic, uni, themes, caption_nums):
        '''
        topic: utf8
        uni: utf8
        themes: utf8
        caption_nums: str
        '''
        if topic.decode('utf-8') in self.word2ix:
            topic_ix = self.word2ix[topic.decode('utf-8')]
        else:
            print 'Topic not in dict'
            return None 
        if uni == '正':
            uni_ix = self.word2ix[uni.decode('utf-8')]
        elif uni == '负':
            uni_ix = self.word2ix[uni.decode('utf-8')]
        else:
            print 'we get wrong tendency:', uni
            return None 
        themes_l = themes.split()
        themes_ix = []
        for word in themes_l:
            if word.decode('utf8') in self.word2ix:
                themes_ix.append(self.word2ix[word.decode('utf8')])
            else:
                print 'themes: %s not in'%word

        caption_nums = int(caption_nums)
        image_feature = np.zeros([caption_nums, self.max_themes])
        for ind in range(caption_nums):
            image_feature[ind, 0] = topic_ix
            image_feature[ind, 1] = uni_ix
            theme_randint = np.min([np.random.randint(self.max_themes-2), len(themes_ix)])
            if themes_ix:
                image_feature[ind, 2:theme_randint+2] = np.random.choice(themes_ix, theme_randint, replace=False)

        return image_feature

    # mscoco
    def get_source_test_for_eval(self):
        image_feature = np.zeros([self.source_num_test_images, self.max_themes])
        image_id = np.zeros([self.source_num_test_images], dtype=int)
        # for i in range(self.source_num_test_images):
            # image_feature[i, :] = self.img_feat[get_key(self.id2filename[self.source_test_images[i]])]
            # image_id[i] = int(self.source_test_images[i])
        for ind, i in enumerate(self.source_test_image_filename):
            image_id[ind] = int(self.source_test_filename2id[i])
            tmp = []
            for x in self.source_test_images[ind]:
                if x in self.word2ix:
                    tmp.append(self.word2ix[x])
                if len(tmp) == self.max_themes:
                    break
            image_feature[ind, :] = np.pad(
                    np.array(tmp),
                    (0, self.max_themes-len(tmp)),
                    'constant',
                    constant_values=(0, 0)
                    )
        return image_feature, image_id, self.source_test_annotation

    def get_wrong_text(self, num_data, phase='train'):
        assert phase=='train'
        idx = range(self.num_train)
        np.random.shuffle(idx)
        caption_train = self.caption_train[idx, :]
        return caption_train[:num_data, :]

    def get_paired_data(self, num_data, phase):
        if phase == 'train':
            caption = self.caption_train
            img_idx = self.caption_idx_train
        else:
            caption = self.caption_test
            img_idx = self.caption_idx_test

        if num_data > 0:
            caption = caption[:num_data, :]
            img_idx = img_idx[:num_data]
        else:
            if phase=='train':
                num_data = self.num_train
            else:
                num_data = self.num_test
           
        image_feature = np.zeros([num_data, self.max_themes])
        for i in range(num_data):
            image_feature[i, :] = self.train_img_feat[i]
        return image_feature, caption, img_idx

    def preprocess(self, caption, lstm_steps):
        caption_padding = sequence.pad_sequences(caption, padding='post', maxlen=lstm_steps)
        return caption_padding

    def decode(self, sent_idx, type='string', remove_END=False):
        if len(sent_idx.shape) == 1:
            sent_idx = np.expand_dims(sent_idx, 0)
        sentences = []
        indexes = []
        for s in range(sent_idx.shape[0]):
            index = []
            sentence = u''.encode('utf8')
            for i in range(sent_idx.shape[1]):
                if int(sent_idx[s][i]) == int(self.word2ix[u'<EOS>']):
                    if not remove_END:
                        #sentence = sentence + '<EOS>'
                        index.append(int(sent_idx[s][i]))
                    break
                else:
                    try:
                        word = self.ix2word[str(int(sent_idx[s][i]))]
                        # print 'index, word:', sent_idx[s][i], 
                        # print word.encode('utf8')
                        sentence = sentence + word.encode('utf8') + u' '.encode('utf8')
                        index.append(int(sent_idx[s][i]))
                    except KeyError:
                        sentence = sentence + u"<UNK>".encode('utf8') + u' '.encode('utf8')
                        index.append(int(self.word2ix[u'<UNK>'.encode('utf8')]))
            indexes.append(index)
            # print 'sentence:', sentence
            sentences.append(sentence + u'.'.encode('utf8'))
        if type=='string':
            # print 'decode:', sentences[-1]
            return sentences
        elif type=='index':
            return indexes

    def flickr_sequential_sample(self, batch_size):
        end = (self.current_flickr+batch_size) % self.num_train_images_flickr
        image_feature = np.zeros([batch_size, self.max_themes])
        if self.current_flickr + batch_size < self.num_train_images_flickr:
            key = self.train_img_idx[self.current_flickr:end]
        else:
            key = np.concatenate((self.train_img_idx[self.current_flickr:], self.train_img_idx[:end]), axis=0)

        count = 0
        img_id_list = []
        for k in key:
            try:
                # print 'data_loader: flickr_sequential_sample'
                # print k, self.train_flickr_img_id[int(k)]
                image_feature[count] = self.train_flickr_img_feat[int(k)]
                img_id_list.append(self.train_flickr_img_id[int(k)])
                count += 1
            except:
                print 'data_loader: flickr_sequential_sample'
                print key
                print k
        self.current_flickr = end
        return image_feature, img_id_list

    def flickr_caption_sequential_sample(self, batch_size):
        end = (self.current_flickr_caption+batch_size) % self.num_flickr_train_caption
        if self.current_flickr_caption + batch_size < self.num_flickr_train_caption:
            caption = self.flickr_caption_train[self.current_flickr_caption:end, :]
        else:
            caption = np.concatenate((self.flickr_caption_train[self.current_flickr_caption:], self.flickr_caption_train[:end]), axis=0)
            self.flickr_random_shuffle()

        self.current_flickr_caption = end
        return caption

    def sequential_sample(self, batch_size):
        end = (self.current+batch_size) % self.num_train
        if self.current + batch_size < self.num_train:
            caption = self.caption_train[self.current:end, :]
            img_idx = self.caption_idx_train[self.current:end]
            image_feature = self.train_img_feat[self.current:end, :]
        else:
            caption = np.concatenate((self.caption_train[self.current:], self.caption_train[:end]), axis=0)
            img_idx = np.concatenate((self.caption_idx_train[self.current:], self.caption_idx_train[:end]), axis=0)
            image_feature = np.concatenate((self.train_img_feat[self.current:], self.train_img_feat[:end]), axis=0)
            self.random_shuffle()

        # image_feature = np.zeros([batch_size, self.img_dims])
        img_id = []
        for i in range(batch_size):
            # image_feature[i, :] = self.img_feat[get_key(img_idx[i])]
            img_id.append(self.filename2id[img_idx[i]])
        self.current = end
        return image_feature, caption, img_id

    def flickr_get_sentiment_label(self, image_id):
        senti = np.zeros([len(image_id), 2])
        for ind, img in enumerate(image_id):
            uni = self.flickr_id2topic[str(int(img))][1]
            # print uni
            if uni == '正'.decode('utf-8'):
                senti[ind, :] = [0, 1]
            elif uni == '负'.decode('utf-8'):
                senti[ind, :] = [1, 0]
            else:
                print 'we get wrong:'
                print self.flickr_id2topic[str(int(img))]
        # print 'flickr_get_sentiment_label:'
        # print senti
        return senti
