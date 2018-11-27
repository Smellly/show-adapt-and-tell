# encode: utf-8
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from highway import *
import copy 
import sys
sys.path.append('./coco_spice/pycocoevalcap/')
# from coco_spice.pycocoevalcap.eval import COCOEvalCap
from eval import COCOEvalCap
# import pdb
import datetime

class SeqGAN():
    def __init__(self, sess, dataset, D_info, conf=None):
        self.sess = sess
        self.model_name = conf.model_name
        self.batch_size = conf.batch_size
        self.max_iter = conf.max_iter
        self.max_to_keep = conf.max_to_keep
        self.is_train = conf.is_train
        # Testing => dropout rate is 0
        if self.is_train:
            self.drop_out_rate = conf.drop_out_rate
        else:
            self.drop_out_rate = 0

        self.num_train = dataset.num_train
        self.G_hidden_size = conf.G_hidden_size         # 512
        self.D_hidden_size = conf.D_hidden_size         # 512
        self.dict_size = dataset.dict_size
        self.max_words = dataset.max_words
        self.dataset = dataset
        # self.img_dims = self.dataset.img_dims
        self.img_dims = self.dataset.max_themes
        self.checkpoint_dir = conf.checkpoint_dir
        self.load_ckpt = conf.load_ckpt
        self.lstm_steps = self.max_words+1
        self.START = self.dataset.word2ix[u'<BOS>']
        self.END = self.dataset.word2ix[u'<EOS>']
        self.UNK = self.dataset.word2ix[u'<UNK>']
        self.NOT = self.dataset.word2ix[u'<NOT>']
        self.method = conf.method
        self.discount = conf.discount
        self.load_pretrain = conf.load_pretrain
        self.filter_sizes = D_info['filter_sizes']
        self.num_filters = D_info['num_filters']
        self.num_filters_total = sum(self.num_filters)
        self.num_classes = D_info['num_classes']
        self.num_domains = 3
        self.num_sentiment = 2
        self.l2_reg_lambda = D_info['l2_reg_lambda']

        
        # D placeholder
        # self.images = tf.placeholder('float32', [self.batch_size, self.img_dims])
        self.images = tf.placeholder('int32', [self.batch_size, self.img_dims], name='d/images')
        self.senti_label = tf.placeholder('int32', [self.batch_size, self.num_sentiment], name='d_senti/senti_label')
        self.right_text = tf.placeholder('int32', [self.batch_size, self.max_words], name='d/right_text')
        self.wrong_text = tf.placeholder('int32', [self.batch_size, self.max_words], name='d/wrong_text')
        self.wrong_length = tf.placeholder('int32', [self.batch_size], name="wrong_length")
        self.right_length = tf.placeholder('int32', [self.batch_size], name="right_length")

        # Domain Classider
        # self.src_images = tf.placeholder('float32', [self.batch_size, self.img_dims])
        # self.tgt_images = tf.placeholder('float32', [self.batch_size, self.img_dims])
        self.src_images = tf.placeholder('int32', [self.batch_size, self.img_dims], name='dc/src_images')
        self.tgt_images = tf.placeholder('int32', [self.batch_size, self.img_dims], name='dc/tgt_images')
        self.src_text = tf.placeholder('int32', [self.batch_size, self.max_words], name='dc/src_text')
        self.tgt_text = tf.placeholder('int32', [self.batch_size, self.max_words], name='dc/tgt_text')
        # Optimizer
        self.G_optim = tf.train.AdamOptimizer(conf.learning_rate)
        # self.D_optim = tf.train.AdamOptimizer(conf.learning_rate)
        # self.T_optim = tf.train.AdamOptimizer(conf.learning_rate)
        # self.Domain_image_optim = tf.train.AdamOptimizer(conf.learning_rate)
        # self.Domain_text_optim = tf.train.AdamOptimizer(conf.learning_rate)
        # self.Senti_classify_optim = tf.train.AdamOptimizer(conf.learning_rate)
        D_info["sentence_length"] = self.max_words
        self.D_info = D_info

        # print 'senti_label:', senti_label.shape

	###################################################
        # Generator                                       #
        ###################################################
	# G placeholder
	state_list, predict_words_list_sample, log_probs_action_picked_list, \
                self.rollout_mask, self.predict_mask = self.generator(name='G', reuse=False)
        # tf.pack is deprecated by tf.stack
	predict_words_sample = tf.stack(predict_words_list_sample)
        self.predict_words_sample = tf.transpose(predict_words_sample, [1,0]) # B,S
	# for testing
	# argmax prediction
        _, predict_words_list_argmax, log_probs_action_picked_list_argmax, _, self.predict_mask_argmax \
                = self.generator_test(name='G', reuse=True)
        # tf.pack is deprecated by tf.stack
        predict_words_argmax = tf.stack(predict_words_list_argmax)
        self.predict_words_argmax = tf.transpose(predict_words_argmax, [1,0]) # B,S
        '''
	rollout = []
	rollout_length = []
	rollout_num = conf.rollout_num
	for i in range(rollout_num):
	    rollout_i, rollout_length_i = self.rollout(predict_words_list_sample, state_list, name="G")      # S*B, S
	    rollout.append(rollout_i)    # R,B,S
	    rollout_length.append(rollout_length_i) # R,B, 1
	   
        # tf.pack is deprecated by tf.stack
	rollout = tf.stack(rollout)      # R,B,S
	rollout = tf.reshape(rollout, [-1, self.max_words])     # R*B,S
        # tf.pack is deprecated by tf.stack
	rollout_length = tf.stack(rollout_length)    # R,B,1
	rollout_length = tf.reshape(rollout_length, [-1, 1])     # R*B, 1
	rollout_length = tf.squeeze(rollout_length)
	rollout_size = self.batch_size * self.max_words * rollout_num
	images_expand = tf.expand_dims(self.images, 1)  # B,1,I
	images_tile = tf.tile(images_expand, [1, self.max_words, 1])    # B,S,I
	images_tile_transpose = tf.transpose(images_tile, [1,0,2])      # S,B,I
	images_tile_transpose = tf.tile(tf.expand_dims(images_tile_transpose, 0), [rollout_num,1,1,1])  #R,S,B,I
	images_reshape = tf.reshape(images_tile_transpose, [-1, self.img_dims]) #R*S*B,I
        # print 'rollout:', rollout.shape
        # print 'self.senti_label:', self.senti_label.shape
        # print 'rollout_length:', rollout_length.shape
        # print 'images_reshape:', images_reshape.shape
        senti_label = tf.tile(self.senti_label, [self.max_words*rollout_num, 1])
        # print 'senti_label:', senti_label.shape
        '''

        self.G_loss = (-1)*tf.reduce_sum(log_probs_action_picked_list)
        ###################################################
        # Record the paramters                            #
        ###################################################
        params = tf.trainable_variables()
        self.R_params = []
        self.G_params = []
        self.D_params = []
        self.G_params_dict = {}
        self.D_params_dict = {}
        for param in params:
            if "R" in param.name:
                self.R_params.append(param)
            elif "G" in param.name:
                self.G_params.append(param)
                self.G_params_dict.update({param.name:param})
            elif "D" in param.name:
                self.D_params.append(param)
                self.D_params_dict.update({param.name:param})
        print "Build graph complete"

    def generator(self, name='generator', reuse=False):
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):
	    if reuse:
                tf.get_variable_scope().reuse_variables()
            # with tf.variable_scope("images"):
            #     # "generator/images"
            #     theme_embedding = tf.get_variable("word_emb_W", [self.img_dims, self.G_hidden_size], "float32", random_uniform_init)
            #     #images_b = tf.get_variable("images_b", [self.G_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.G_hidden_size, state_is_tuple=True)
                lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, output_keep_prob=1-self.drop_out_rate)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                # "generator/embedding"
                word_emb_W = tf.get_variable("word_emb_W", [self.dict_size, self.G_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("output"):
                # "generator/output"
                # dict size minus 1 => remove <UNK>
                output_W = tf.get_variable("output_W", [self.G_hidden_size, self.dict_size], "float32", random_uniform_init)

            start_token = tf.constant(self.START, dtype=tf.int32, shape=[self.batch_size])
            state = lstm1.zero_state(self.batch_size, 'float32')
	    mask = tf.constant(True, "bool", [self.batch_size])
	    log_probs_action_picked_list = []
	    predict_words = []
	    state_list = []
	    predict_mask_list = []
            for j in range(self.max_words+1):
                # print 'j:', j
                if j == 0:
		    # images_emb = tf.matmul(self.images, theme_embedding)
                    with tf.variable_scope("images"):
                        with tf.device("/cpu:0"):
                            images_emb = tf.nn.embedding_lookup(word_emb_W, self.images)
                    # print 'images shape:', self.images.shape
                    # print 'images_emb shape:', images_emb.shape
                    lstm1_in = tf.reduce_sum(images_emb, 1)
                    # print 'lstm1_in shape:', lstm1_in.shape
                else:
                    tf.get_variable_scope().reuse_variables()
                    with tf.device("/cpu:0"):
                        if j == 1:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, tf.stop_gradient(sample_words))
                with tf.variable_scope("lstm"):
                    # "generator/lstm"
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())     # output: B,H
                if j > 0:
		    logits = tf.matmul(output, output_W)
		    log_probs = tf.log(tf.nn.softmax(logits)+1e-8)	# B,D
		    # word drawn from the multinomial distribution
		    sample_words = tf.reshape(tf.multinomial(log_probs,1), [self.batch_size])
                    # tf.select is deprecated after tensorflow 0.12
		    # mask_out_word = tf.select(mask, sample_words,
		    mask_out_word = tf.where(mask, sample_words,
						tf.constant(self.NOT, dtype=tf.int64, shape=[self.batch_size]))
                    predict_words.append(mask_out_word)
		    #predict_words.append(sample_words)
		    # the mask should be dynamic
		    # if the sentence is: This is a dog <END>
		    # the predict_mask_list is: 1,1,1,1,1,0,0,.....
		    predict_mask_list.append(tf.to_float(mask))
		    action_picked = tf.range(self.batch_size)*(self.dict_size) + tf.to_int32(sample_words)        # B
		    # mask out the word beyond the <END>
                    # tf.mul is deprecated
		    log_probs_action_picked = tf.multiply(
                            tf.gather(tf.reshape(log_probs, [-1]), action_picked), tf.to_float(mask))
                    log_probs_action_picked_list.append(log_probs_action_picked)
                    prev_mask = mask
                    mask_step = tf.not_equal(sample_words, self.END)    # B
                    mask = tf.logical_and(prev_mask, mask_step)
                    state_list.append(state)

            # tf.pack is deprecated by tf.stack
	    # predict_mask_list = tf.pack(predict_mask_list)      # S,B
	    predict_mask_list = tf.stack(predict_mask_list)      # S,B
            predict_mask_list = tf.transpose(predict_mask_list, [1,0])  # B,S
            log_probs_action_picked_list = tf.stack(log_probs_action_picked_list)        # S,B
            log_probs_action_picked_list = tf.reshape(log_probs_action_picked_list, [-1])       # S*B
            return state_list, predict_words, log_probs_action_picked_list, None, predict_mask_list

    def generator_test(self, name='generator', reuse=False):
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # with tf.variable_scope("images"):
            #     # "generator/images"
            #     theme_embedding = tf.get_variable(
            #             "word_emb_W", [self.img_dims, self.G_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.G_hidden_size, state_is_tuple=True)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                # "generator/embedding"
                word_emb_W = tf.get_variable(
                        "word_emb_W", [self.dict_size, self.G_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("output"):
                # "generator/output"
                # dict size minus 1 => remove <UNK>
                output_W = tf.get_variable(
                        "output_W", [self.G_hidden_size, self.dict_size], "float32", random_uniform_init)
            start_token = tf.constant(self.START, dtype=tf.int32, shape=[self.batch_size])
            state = lstm1.zero_state(self.batch_size, 'float32')
            mask = tf.constant(True, "bool", [self.batch_size])
            log_probs_action_picked_list = []
            predict_words = []
            state_list = []
            predict_mask_list = []
            for j in range(self.max_words+1):
                if j == 0:
                    # images_emb = tf.matmul(self.images, theme_embedding)
                    # with tf.variable_scope("images"):
                    with tf.device("/cpu:0"):
                        images_emb = tf.nn.embedding_lookup(word_emb_W, self.images)
                    lstm1_in = tf.reduce_sum(images_emb, 1)
                else:
                    tf.get_variable_scope().reuse_variables()
                    with tf.device("/cpu:0"):
                        if j == 1:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, tf.stop_gradient(sample_words))
                with tf.variable_scope("lstm"):
                    # "generator/lstm"
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())     # output: B,H
                if j > 0:
                    #logits = tf.matmul(output, output_W) + output_b       # B,D
                    logits = tf.matmul(output, output_W)
                    log_probs = tf.log(tf.nn.softmax(logits)+1e-8)      # B,D
                    # word drawn from the multinomial distribution
                    sample_words = tf.argmax(log_probs, 1)      # B
                    # tf.select is deprecated after tensorflow 0.12
                    mask_out_word = tf.where(mask, sample_words,
                                                tf.constant(self.NOT, dtype=tf.int64, shape=[self.batch_size]))
                    predict_words.append(mask_out_word)
                    # the mask should be dynamic
                    # if the sentence is: This is a dog <END>
                    # the predict_mask_list is: 1,1,1,1,1,0,0,.....
                    predict_mask_list.append(tf.to_float(mask))
                    action_picked = tf.range(self.batch_size)*(self.dict_size) + tf.to_int32(sample_words)        # B
                    # mask out the word beyond the <END>
                    # tf.mul is deprecated
                    log_probs_action_picked = tf.multiply(
                            tf.gather(tf.reshape(log_probs, [-1]), action_picked), tf.to_float(mask))
                    log_probs_action_picked_list.append(log_probs_action_picked)
                    prev_mask = mask
                    mask_step = tf.not_equal(sample_words, self.END)    # B
                    mask = tf.logical_and(prev_mask, mask_step)
                    state_list.append(state)

            # tf.pack is deprecated by tf.stack
            predict_mask_list = tf.stack(predict_mask_list)      # S,B
            predict_mask_list = tf.transpose(predict_mask_list, [1,0])  # B,S
            # tf.pack is deprecated by tf.stack
            log_probs_action_picked_list = tf.stack(log_probs_action_picked_list)        # S,B
            log_probs_action_picked_list = tf.reshape(log_probs_action_picked_list, [-1])       # S*B
            return state_list, predict_words, log_probs_action_picked_list, None, predict_mask_list

    def rollout(self, predict_words, state_list, name="R"):
	random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)	
	with tf.variable_scope(name):
	    tf.get_variable_scope().reuse_variables()
	    # with tf.variable_scope("images"):
            #     # "generator/images"
            #     theme_embedding = tf.get_variable("word_emb_W", [self.img_dims, self.G_hidden_size], "float32", random_uniform_init)
	    with tf.variable_scope("lstm"):
                # WONT BE CREATED HERE
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.G_hidden_size, state_is_tuple=True)
		lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, output_keep_prob=1-self.drop_out_rate)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                # "R/embedding"
                word_emb_W = tf.get_variable("word_emb_W",[self.dict_size, self.G_hidden_size], "float32", random_uniform_init)
            with tf.variable_scope("output"):
                # "R/output"
                output_W = tf.get_variable("output_W", [self.G_hidden_size, self.dict_size], "float32", random_uniform_init)
	    rollout_list = []
	    length_mask_list = []
	    # rollout for the first time step
	    for step in range(self.max_words):
		sample_words = predict_words[step]
		state = state_list[step]
		rollout_step_list = []
		mask = tf.constant(True, "bool", [self.batch_size])		
		# used to calcualte the length of the rollout sentence
		length_mask_step = []
		for j in range(step+1):
                    # tf.select is deprecated after tensorflow 0.12
		    mask_out_word = tf.where(mask, predict_words[j], 
						tf.constant(self.NOT, dtype=tf.int64, shape=[self.batch_size]))
		    rollout_step_list.append(mask_out_word)
		    length_mask_step.append(mask)
		    prev_mask = mask
                    mask_step = tf.not_equal(predict_words[j], self.END)    # B
                    mask = tf.logical_and(prev_mask, mask_step)
		for j in range(self.max_words-step-1):
		    if step != 0 or j != 0:
	                tf.get_variable_scope().reuse_variables()
            	    with tf.device("/cpu:0"):
                	sample_words_emb = tf.nn.embedding_lookup(word_emb_W, tf.stop_gradient(sample_words))
  	            with tf.variable_scope("lstm"):
                	output, state = lstm1(sample_words_emb, state, scope=tf.get_variable_scope())     # output: B,H
		    logits = tf.matmul(output, output_W)
		    # add 1e-8 to prevent log(0) 
            	    log_probs = tf.log(tf.nn.softmax(logits)+1e-8)   # B,D
            	    sample_words = tf.squeeze(tf.multinomial(log_probs,1))
                    # tf.select is deprecated after tensorflow 0.12
                    # tf.where(condition, x, y, name)
                    # The condition tensor acts as a mask that chooses, 
                    # based on the value at each element, 
                    # whether the corresponding element / row in the output 
                    # should be taken from x (if true) or y (if false).
		    mask_out_word = tf.where(mask, sample_words, 
						tf.constant(self.NOT, dtype=tf.int64, shape=[self.batch_size]))
		    rollout_step_list.append(mask_out_word)
		    length_mask_step.append(mask)
                    prev_mask = mask
                    mask_step = tf.not_equal(sample_words, self.END)    # B
                    mask = tf.logical_and(prev_mask, mask_step)

                # tf.pack is deprecated by tf.stack
		length_mask_step = tf.stack(length_mask_step)	# S,B
		length_mask_step = tf.transpose(length_mask_step, [1,0])	# B,S
		length_mask_list.append(length_mask_step)
                # tf.pack is deprecated by tf.stack
		rollout_step_list = tf.stack(rollout_step_list)	# S,B
		rollout_step_list = tf.transpose(rollout_step_list, [1,0])	# B,S
		rollout_list.append(rollout_step_list)
	
            # tf.pack is deprecated by tf.stack
	    length_mask_list = tf.stack(length_mask_list)	# S,B,S
	    length_mask_list = tf.reshape(length_mask_list, [-1, self.max_words])	# S*B,S
            # tf.pack is deprecated by tf.stack
	    rollout_list = tf.stack(rollout_list)	# S,B,S
	    rollout_list = tf.reshape(rollout_list, [-1, self.max_words])	# S*B, S
	    rollout_length = tf.to_int32(tf.reduce_sum(tf.to_float(length_mask_list),1))
 	    return rollout_list, rollout_length

    def train(self):
        print "--------------------------model train---------------------------"
        self.G_train_op = self.G_optim.minimize(self.G_loss, var_list=self.G_params)
        # self.G_hat_train_op = self.T_optim.minimize(self.teacher_loss, var_list=self.G_params)
        self.D_train_op = self.D_optim.minimize(self.D_loss, var_list=self.D_params)
        self.Domain_text_train_op = self.Domain_text_optim.minimize(self.D_text_loss)
        self.Senti_classify_train_op = self.Senti_classify_optim.minimize(self.D_senti_loss)

        log_dir = os.path.join('.', 'logs_m', self.model_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = tf.summary.FileWriter(os.path.join(log_dir, "SeqGAN_sample"), self.sess.graph)
        # self.summary_op = tf.summary.merge_all()
        tf.initialize_all_variables().run()
        if self.load_pretrain:
            print "[@] Load the pretrained model %s."%self.load_ckpt
            self.G_saver = tf.train.Saver(self.G_params_dict)
            # self.G_saver.restore(self.sess, "./checkpoint/mscoco/G_pretrained/G_Pretrained-39000")
            self.G_saver.restore(self.sess, self.load_ckpt)

        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        count = 0
        D_count = 0
        G_count = 0
        freq_to_eval = 200
        for idx in range(self.max_iter//freq_to_eval):
            self.save(self.checkpoint_dir, count)
            print "Epoch    : %d"%(idx)
            print "Iter     : %d"%(count)
            print "G_iter   : %d"%(G_count)
            print "D_iter   : %d"%(D_count)
            self.evaluate(count)
            for _ in tqdm(range(freq_to_eval)):
                tgt_image_feature, image_id = self.dataset.flickr_sequential_sample(self.batch_size)
                senti_label = self.dataset.flickr_get_sentiment_label(image_id)
                tgt_text = self.dataset.flickr_caption_sequential_sample(self.batch_size)
                image_feature, right_text, _ = self.dataset.sequential_sample(self.batch_size)
                nonENDs = np.array(map(lambda x: (x != self.NOT).sum(), right_text))
                mask_t = np.zeros([self.batch_size, self.max_words])
                for ind, row in enumerate(mask_t):
                    # mask out the <BOS>
                    row[0:nonENDs[ind]] = 1

                wrong_text = self.dataset.get_wrong_text(self.batch_size)
                # print 'right_text shape:', right_text.shape
                # print 'wroong_text shape:', wrong_text.shape
                # print 'right_text:', right_text[0, :]
                # print 'wroong_text:', wrong_text[0, :]
                right_length = np.sum((right_text!=self.NOT)+0, 1)
                wrong_length = np.sum((wrong_text!=self.NOT)+0, 1)
                for _ in range(1):      # g_step
                    # update G
                    feed_dict = {
                            self.images: tgt_image_feature,
                            self.senti_label: senti_label,
                        }
                    _, G_loss, g_summary = self.sess.run([self.G_train_op, self.G_loss, self.G_summary], feed_dict)
                    self.writer.add_summary(g_summary, G_count)
                    G_count += 1
                for _ in range(5):      # d_step    
                    # update D
                    feed_dict = {self.images: image_feature, 
                        self.right_text:right_text, 
                        self.wrong_text:wrong_text, 
                        self.right_length:right_length,
                        self.wrong_length:wrong_length,
                        self.mask: mask_t,
                        self.src_images: image_feature,
                        self.tgt_images: tgt_image_feature,
                        self.src_text: right_text,
                        self.tgt_text: tgt_text
                        }
                    _, D_loss, d_summary = self.sess.run(
                                [self.D_train_op, self.D_loss, self.D_summary], 
                                feed_dict
                            )
                    _, D_text_loss, d_text_summary = self.sess.run(
                            [self.Domain_text_train_op, self.D_text_loss, self.D_text_summary],
                            {
                                self.src_text: right_text,
                                self.tgt_text: tgt_text,
                                self.images: tgt_image_feature
                                }
                            )
                    _, D_senti_loss, d_senti_summary = self.sess.run(
                            [self.Senti_classify_train_op, self.D_senti_loss_sum, self.D_senti_summary],
                            {
                                self.images: tgt_image_feature,
                                self.tgt_text: tgt_text,
                                self.senti_label: senti_label
                                }
                            )
                    self.writer.add_summary(d_summary, D_count)
                    self.writer.add_summary(d_text_summary, D_count)
                    self.writer.add_summary(d_senti_summary, D_count)
                    D_count += 1
                # summary_str = self.sess.run([self.summary_op]) 
                # self.writer.add_summary(summary_str, count)
                count += 1

    def evaluate(self, count, score_verbose=True, output_path=None):
        oldtime = datetime.datetime.now()
        print 'TIME:', oldtime
        samples = []
        print 'loading all the topics'
        samples_index = []
        image_feature, image_id, test_annotation = self.dataset.get_test_for_eval()
        num_samples = image_feature.shape[0] # self.dataset.source_num_test_images
        samples_index = np.full([self.batch_size*(num_samples//self.batch_size), self.max_words], self.NOT)
        # print 'image_feature size:', image_feature.shape
        print 'num_samples:', num_samples
        for i in range(num_samples//self.batch_size):
            image_feature_feed = np.zeros([self.batch_size, self.dataset.max_themes])
            image_feature_test = image_feature[i*self.batch_size:(i+1)*self.batch_size]
            image_feature_test_length = len(image_feature_test)
            # if image_feature_test_length < self.batch_size:
            #     image_feature_test = image_feature[-self.batch_size]
            for j in range(image_feature_test_length):
                image_feature_feed[j, :] = image_feature_test[j]
            feed_dict = {self.images: image_feature_feed}
            predict_words = self.sess.run(self.predict_words_argmax, feed_dict)
            for j in range(self.batch_size - image_feature_test_length, self.batch_size):
                samples.append([self.dataset.decode(predict_words[j, :], type='string', remove_END=True)[0]])
                sample_index = self.dataset.decode(predict_words[j, :], type='index', remove_END=False)[0]
                samples_index[i*self.batch_size+j][:len(sample_index)] = sample_index
        # predict from samples
        samples = np.asarray(samples)
        samples_index = np.asarray(samples_index)
        output = []
        for ii in range(200):
            tmp = u''
            for i in image_feature[ii, :]:
                tmp += self.dataset.ix2word[str(int(i))] + u' '
            print '[%] Topic : ', tmp.encode('utf-8') 
            print '[-] Sentence:', samples[ii][0]
            print '[%] GroundTruth:', test_annotation[str(image_id[ii])][0]['caption'].encode('utf8')
            print '[%] TIME:', datetime.datetime.now()
            output.append(tmp.encode('utf-8') + '# ' + samples[ii][0] + '\n')
        if output_path:
            with open(output_path, 'w') as f:
                f.writelines(output)
        meteor_pd = {}
        meteor_id = []
        for j in range(len(samples)):
            if image_id[j] == 0:
                break
            meteor_pd[str(int(image_id[j]))] = [{'image_id':str(int(image_id[j])), 'caption':samples[j][0]}]
            meteor_id.append(str(int(image_id[j])))
        sample_dir = os.path.join("./SeqGAN_samples_sample", self.model_name)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        file_name = "%s_%s" % (self.dataset.dataset_name, str(count))
        np.savez(os.path.join(sample_dir, file_name), string=samples, index=samples_index, id=meteor_id)
        newtime = datetime.datetime.now()
        print 'TIME:', newtime
        print 'lap:', (newtime-oldtime).seconds
        print 'speed: %f second per tweet'%(shint((newtime-oldtime).seconds)/num_samples)
        if score_verbose:
            scorer = COCOEvalCap(test_annotation, meteor_pd, meteor_id)
            # scorer.evaluate(verbose=True)
            scorer.evaluate()

    def save(self, checkpoint_dir, step):
        model_name = "SeqGAN_sample"
        model_dir = "%s" % (self.dataset.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir, self.model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s" % (self.dataset.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, topic, tendency, themes, caption_nums):
        print "--------------------------model inference---------------------------"
        self.G_train_op = self.G_optim.minimize(self.G_loss, var_list=self.G_params)
        # tf.initialize_all_variables().run()
        tf.global_variables_initializer()
        if self.load_pretrain:
            print "[@] Load the pretrained model %s."%self.load_ckpt
            self.G_saver = tf.train.Saver(self.G_params_dict)
            self.G_saver.restore(self.sess, self.load_ckpt)

        output_path = '/home/smelly/projects/show-adapt-and-tell/topic-adapt-tell/outputtest.txt'

        oldtime = datetime.datetime.now()
        print 'TIME:', oldtime
        samples = []
        print 'loading all the topics'
        samples_index = []
        image_feature = self.dataset.get_specific_for_eval(topic, tendency, themes, caption_nums)
        num_samples = image_feature.shape[0] # self.dataset.source_num_test_images
        samples_index = np.full([self.batch_size*(num_samples//self.batch_size), self.max_words], self.NOT)
        # print 'image_feature size:', image_feature.shape
        print 'num_samples:', num_samples
        if self.batch_size > num_samples:
            print 'batch size %d is larger than num_samples %d'%(self.batch_size, num_samples)
            num_samples = self.batch_size
        for i in range(num_samples//self.batch_size):
            image_feature_feed = np.zeros([self.batch_size, self.dataset.max_themes])
            image_feature_test = image_feature[i*self.batch_size:(i+1)*self.batch_size]
            image_feature_test_length = len(image_feature_test)
            # if image_feature_test_length < self.batch_size:
            #     image_feature_test = image_feature[-self.batch_size]
            for j in range(image_feature_test_length):
                image_feature_feed[j, :] = image_feature_test[j]
            feed_dict = {self.images: image_feature_feed}
            predict_words = self.sess.run(self.predict_words_argmax, feed_dict)
            # print 'predict_words:', predict_words
            for j in range(self.batch_size - image_feature_test_length, self.batch_size):
                samples.append([self.dataset.decode(predict_words[j, :], type='string', remove_END=True)[0]])
                sample_index = self.dataset.decode(predict_words[j, :], type='index', remove_END=False)[0]
                samples_index[i*self.batch_size+j][:len(sample_index)] = sample_index
        # predict from samples
        if not samples:
            print 'something wrong'
            return None
        samples = np.asarray(samples)
        samples_index = np.asarray(samples_index)
        output = []
        for ii in range(int(caption_nums)):
            tmp = u''
            for i in image_feature[ii, :]:
                tmp += self.dataset.ix2word[str(int(i))] + u' '
            print '[%] Topic : ', tmp.encode('utf-8') 
            print '[-] Sentence:', samples[ii][0]
            print '[%] TIME:', datetime.datetime.now()
            output.append(tmp.encode('utf-8') + '# ' + samples[ii][0] + '\n')
        if output_path:
            with open(output_path, 'w') as f:
                f.writelines(output)
