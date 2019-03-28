#coding=utf-8
'''
Author: Chence Shi
Contact: chenceshi@pku.edu.cn
'''

import tensorflow as tf 
import sys
import os

import numpy as np 


def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


class FOSSIL(object):
    def __init__(self, args, n_items, n_users):
        self.args = args
        self.n_items = n_items
        self.n_users = n_users
        self._build()

        self.saver = tf.train.Saver()

    def _build(self):
        self.inp = tf.placeholder(tf.int32, shape=(None, None), name='inp') # if maxlen is 5, valid len of sample i is 3, then inp[i] = [0, 0, x, x, x]
        self.user = tf.placeholder(tf.int32, shape=(None), name='user')
        self.pos = tf.placeholder(tf.int32, shape=(None), name='pos')
        self.neg = tf.placeholder(tf.int32, shape=(None, self.args.neg_size), name='neg')  

        self.lr = tf.placeholder(tf.float32, shape=None, name='lr')

        self.item_embedding1 = tf.get_variable('item_embedding1', 
                                               shape=(self.n_items+1, self.args.emsize),
                                               dtype=tf.float32,
                                               regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg),
                                               initializer=tf.truncated_normal_initializer(stddev=0.01))
        self.item_embedding2 = tf.get_variable('item_embedding2', 
                                               shape=(self.n_items+1, self.args.emsize),
                                               dtype=tf.float32,
                                               regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg),
                                               initializer=tf.truncated_normal_initializer(stddev=0.01))     
        self.user_bias = tf.get_variable('user_bias', 
                                               shape=(self.n_users+1, self.args.order),
                                               dtype=tf.float32,
                                               regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg),
                                               initializer=tf.truncated_normal_initializer(stddev=0.01))
        self.global_bias = tf.get_variable('global_bias', 
                                               shape=(self.args.order),
                                               dtype=tf.float32,
                                               regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg),
                                               initializer=tf.constant_initializer(0.))
        self.item_bias = tf.get_variable('item_bias', 
                                               shape=(self.n_items+1),
                                               dtype=tf.float32,
                                               regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg),
                                               initializer=tf.constant_initializer(0.))        

        mask_inp = tf.expand_dims(tf.to_float(tf.not_equal(self.inp, 0)), -1) #(batch, maxlen, 1)
        len_inp = tf.reduce_sum(tf.squeeze(mask_inp, axis=2), axis=1) #(batch)
        item_embed = tf.nn.embedding_lookup(self.item_embedding1, self.inp) * mask_inp #(batch, maxlen, k)
        long_term = tf.reduce_sum(item_embed, axis=1) #(batch, k)
        long_term = tf.expand_dims(tf.pow(len_inp, -self.args.alpha), -1) * long_term #(batch, k)

        effective_order = tf.minimum(len_inp, self.args.order) #(batch)
        effective_order = tf.expand_dims(tf.to_float(tf.sequence_mask(effective_order,self.args.order)), -1) #(batch, order, 1)

        short_term = tf.nn.embedding_lookup(self.user_bias, self.user) #(batch, order)
        short_term = tf.expand_dims(short_term + self.global_bias, axis=-1) #(batch, order, 1)
        short_term = short_term * item_embed[:, :-1-self.args.order:-1]  #(batch, order, k)
        short_term = tf.reduce_sum(short_term * effective_order, axis=1) #(batch, k)

        ### for train only
        pos_bias = tf.nn.embedding_lookup(self.item_bias, self.pos) #(batch)
        pos_embed = tf.nn.embedding_lookup(self.item_embedding2, self.pos) #(batch, k)
        neg_bias = tf.nn.embedding_lookup(self.item_bias, self.neg) #(batch, neg_size)
        neg_embed = tf.nn.embedding_lookup(self.item_embedding2, self.neg) #(batch, neg_size, k)

        temp_vec = short_term + long_term #(batch, k)

        pos_score = pos_bias + tf.reduce_sum(temp_vec*pos_embed, axis=1) #(batch)
        neg_score = neg_bias + tf.reduce_sum(tf.expand_dims(temp_vec, axis=1) * neg_embed, axis=2) #(batch, neg_size)
        neg_score = tf.reduce_mean(neg_score, axis=1) #(batch)

        loss = -tf.reduce_mean(tf.log(tf.clip_by_value(tf.sigmoid(pos_score-neg_score), 1e-24, 1-1e-24)))

        ### for prediction only
        full_score = self.item_bias + tf.matmul(temp_vec, self.item_embedding2, transpose_b=True) #(batch, n_items+1)
        self.prediction = full_score
        self.loss = loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)
        if self.args.optim == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.args.optim == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        else:
            raise NotImplementedError

        self.train_op = self.optimizer.minimize(self.loss)
        self.recall_at_k, self.ndcg_at_k = self._metric_at_k()

    def _metric_at_k(self, k=20):
        prediction = self.prediction #(batch, n_items+1)
        prediction_transposed = tf.transpose(prediction)
        labels = tf.reshape(self.pos, shape=(-1,))
        pred_values = tf.expand_dims(tf.diag_part(tf.nn.embedding_lookup(prediction_transposed, labels)), -1)
        tile_pred_values = tf.tile(pred_values, [1, self.n_items])
        ranks = tf.reduce_sum(tf.cast(prediction[:,1:] > tile_pred_values, dtype=tf.float32), -1) + 1
        ndcg = 1. / (log2(1.0 + ranks))
        hit_at_k = tf.nn.in_top_k(prediction, labels, k=k)
        hit_at_k = tf.cast(hit_at_k, dtype=tf.float32)
        ndcg_at_k = ndcg * hit_at_k
        return tf.reduce_sum(hit_at_k), tf.reduce_sum(ndcg_at_k)





class FPMC(object):
    def __init__(self, args, n_items, n_users):
        self.args = args
        self.n_items = n_items
        self.n_users = n_users
        self._build()

        self.saver = tf.train.Saver()

    def _build(self):
        self.inp = tf.placeholder(tf.int32, shape=(None, None), name='inp') # if maxlen is 5, valid len of sample i is 3, then inp[i] = [0, 0, x, x, x]
        self.user = tf.placeholder(tf.int32, shape=(None), name='user')
        self.pos = tf.placeholder(tf.int32, shape=(None), name='pos')
        self.neg = tf.placeholder(tf.int32, shape=(None, self.args.neg_size), name='neg')  

        self.lr = tf.placeholder(tf.float32, shape=None, name='lr')


        self.VUI = tf.get_variable('user_item', 
                                               shape=(self.n_users+1, self.args.emsize),
                                               dtype=tf.float32,
                                               regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg),
                                               initializer=tf.truncated_normal_initializer(stddev=0.01))
        self.VIU = tf.get_variable('item_user', 
                                               shape=(self.n_items+1, self.args.emsize),
                                               dtype=tf.float32,
                                               regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg),
                                               initializer=tf.truncated_normal_initializer(stddev=0.01))     
        self.VIL = tf.get_variable('item_prev', 
                                               shape=(self.n_items+1, self.args.emsize),
                                               dtype=tf.float32,
                                               regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg),
                                               initializer=tf.truncated_normal_initializer(stddev=0.01))
        self.VLI = tf.get_variable('prev_item', 
                                               shape=(self.n_items+1, self.args.emsize),
                                               dtype=tf.float32,
                                               regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg),
                                               initializer=tf.truncated_normal_initializer(stddev=0.01))


        self.prev = self.inp[:, -1] #(batch)
        u = tf.nn.embedding_lookup(self.VUI, self.user) #(batch, k)
        prev = tf.nn.embedding_lookup(self.VLI, self.prev) #(batch, k)     

        ### for train only
        pos_iu = tf.nn.embedding_lookup(self.VIU, self.pos) #(batch, k)
        pos_il = tf.nn.embedding_lookup(self.VIL, self.pos) #(batch, k)
        pos_score = tf.reduce_sum(u*pos_iu, axis=1) + tf.reduce_sum(prev*pos_il, axis=1) #(batch)

        neg_iu = tf.nn.embedding_lookup(self.VIU, self.neg) #(batch, neg, k)
        neg_il = tf.nn.embedding_lookup(self.VIL, self.neg) #(batch, neg, k)
        neg_score = tf.reduce_sum(tf.expand_dims(u, 1)*neg_iu, axis=2) + tf.reduce_sum(tf.expand_dims(prev, 1)*neg_il, axis=2) #(batch, neg)
        neg_score = tf.reduce_mean(neg_score, axis=1) #(batch)

        loss = -tf.reduce_mean(tf.log(tf.clip_by_value(tf.sigmoid(pos_score-neg_score), 1e-24, 1-1e-24)))

        ### for prediction only
        full_score = tf.matmul(u, self.VIU, transpose_b=True) + tf.matmul(prev, self.VIL, transpose_b=True)  #(batch, n_items+1)

        self.prediction = full_score
        self.loss = loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)
        if self.args.optim == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.args.optim == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        else:
            raise NotImplementedError

        self.train_op = self.optimizer.minimize(self.loss)
        self.recall_at_k, self.ndcg_at_k = self._metric_at_k()


    def _metric_at_k(self, k=20):
        prediction = self.prediction #(batch, n_items+1)
        prediction_transposed = tf.transpose(prediction)
        labels = tf.reshape(self.pos, shape=(-1,))
        pred_values = tf.expand_dims(tf.diag_part(tf.nn.embedding_lookup(prediction_transposed, labels)), -1)
        tile_pred_values = tf.tile(pred_values, [1, self.n_items])
        ranks = tf.reduce_sum(tf.cast(prediction[:,1:] > tile_pred_values, dtype=tf.float32), -1) + 1
        ndcg = 1. / (log2(1.0 + ranks))
        hit_at_k = tf.nn.in_top_k(prediction, labels, k=k)
        hit_at_k = tf.cast(hit_at_k, dtype=tf.float32)
        ndcg_at_k = ndcg * hit_at_k
        return tf.reduce_sum(hit_at_k), tf.reduce_sum(ndcg_at_k)











