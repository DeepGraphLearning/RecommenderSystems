#coding=utf-8
'''
Author: Weiping Song
Contact: songweiping@pku.edu.cn
'''

import tensorflow as tf
import sys
from .base import LSTMNet
from .base import TemporalConvNet
from .base import TransformerNet

def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

class NeuralSeqRecommender(object):
    def __init__(self, args, n_items, n_users):
        self.args = args
        self.n_items = n_items
        self.n_users = n_users
        self._build()

        self.saver = tf.train.Saver()

    def _build(self):
        self.inp = tf.placeholder(tf.int32, shape=(None, None), name='inp')
        self.pos = tf.placeholder(tf.int32, shape=(None, None), name='pos')
        self.neg = tf.placeholder(tf.int32, shape=(None, None, self.args.neg_size), name='neg')

        self.lr = tf.placeholder(tf.float32, shape=None, name='lr')
        self.dropout = tf.placeholder_with_default(0., shape=())
        self.item_embedding = item_embedding = tf.get_variable('item_embedding', \
                                shape=(self.n_items + 1, self.args.emsize), \
                                dtype=tf.float32, \
                                regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg), \
                                initializer=tf.contrib.layers.xavier_initializer())

        input_item = tf.nn.embedding_lookup(item_embedding, self.inp)
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inp, 0)), -1)

        if self.args.model == 'tcn':
            num_channels = [self.args.nhid] * (self.args.levels -1 ) + [self.args.emsize]
            self.net = TemporalConvNet(num_channels, stride=1, kernel_size=self.args.ksize, dropout=self.dropout)
        elif self.args.model == 'rnn':
            self.net = LSTMNet(layers=self.args.levels, hidden_units=self.args.nhid, dropout=self.dropout)
        elif self.args.model == 'transformer':
            self.net = TransformerNet(self.args.emsize, self.args.levels, self.args.num_heads, self.args.seq_len, dropout_rate=self.dropout, pos_fixed=self.args.pos_fixed)
        else:
            raise NotImplementedError

        outputs = self.net(input_item, mask)
        outputs *= mask
        ct_vec = tf.reshape(outputs, (-1, self.args.emsize))
        outputs_shape = tf.shape(outputs)

        self.total_loss = 0.

        self.istarget = istarget = tf.reshape(tf.to_float(tf.not_equal(self.pos, 0)), [-1])

        _pos_emb = tf.nn.embedding_lookup(self.item_embedding, self.pos)
        pos_emb = tf.reshape(_pos_emb, (-1, self.args.emsize))
        _neg_emb = tf.nn.embedding_lookup(self.item_embedding, self.neg)
        neg_emb = tf.reshape(_neg_emb, (-1, self.args.neg_size, self.args.emsize))
        
        temp_vec_neg = tf.tile(tf.expand_dims(ct_vec, [1]), [1, self.args.neg_size, 1]) 
            
        if self.args.loss == 'ns':
            assert self.args.neg_size == 1
            pos_logit = tf.reduce_sum(ct_vec * pos_emb, -1)
            neg_logit = tf.squeeze(tf.reduce_sum(temp_vec_neg * neg_emb, -1), 1)
            loss = tf.reduce_sum(
                        -tf.log(tf.sigmoid(pos_logit) + 1e-24) * istarget - \
                        tf.log(1 - tf.sigmoid(neg_logit) + 1e-24) * istarget \
                    ) / tf.reduce_sum(istarget)
        elif self.args.loss == 'sampled_sm':
            pos_logit = tf.reduce_sum(ct_vec * pos_emb, -1, keepdims=True)
            neg_logit = tf.reduce_sum(temp_vec_neg * neg_emb, -1)
            label_1 = tf.ones_like(pos_logit, dtype=tf.float32)
            label_0 = tf.zeros_like(neg_logit, dtype=tf.float32)
            labels = tf.concat([label_1, label_0], -1)
            logit = tf.concat([pos_logit, neg_logit], -1)
            softmax_logit = tf.nn.softmax(logit)
            loss = tf.reduce_sum( \
                    tf.reduce_sum( \
                    - labels * tf.log(softmax_logit + 1e-24) - \
                    (1. - labels) * tf.log(1. - softmax_logit + 1e-24), -1) * istarget \
                    ) / tf.reduce_sum(istarget)
        elif self.args.loss == 'full_sm':
            full_logits = tf.matmul(ct_vec, self.item_embedding, transpose_b=True)
            loss =  tf.reduce_sum( \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.pos, [-1]), \
                                                                logits=full_logits) * istarget \
                    ) / tf.reduce_sum(istarget)
         
        full_logits = tf.matmul(ct_vec, self.item_embedding, transpose_b=True)
        self.prediction = full_logits
        
        self.loss = loss
        self.total_loss += loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss += sum(reg_losses)

        optimizer = tf.train.AdamOptimizer(self.lr)
        gvs = optimizer.compute_gradients(self.total_loss)
        capped_gvs = [(tf.clip_by_value(grad, -self.args.clip, self.args.clip), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)
        self.hit_at_k, self.ndcg_at_k, self.num_target = self._metric_at_k()


    def _metric_at_k(self, k=20):
        prediction = self.prediction
        prediction_transposed = tf.transpose(prediction)
        labels = tf.reshape(self.pos, shape=(-1,))
        pred_values = tf.expand_dims(tf.diag_part(tf.nn.embedding_lookup(prediction_transposed, labels)), -1)
        tile_pred_values = tf.tile(pred_values, [1, self.n_items])
        ranks = tf.reduce_sum(tf.cast(prediction[:,1:] > tile_pred_values, dtype=tf.float32), -1) + 1

        istarget = tf.reshape(self.istarget, shape=(-1,))
        ndcg = 1. / (log2(1.0 + ranks))
        hit_at_k = tf.nn.in_top_k(prediction, labels, k=k) # also known as Recall@k
        hit_at_k = tf.cast(hit_at_k, dtype=tf.float32)
        istarget = tf.reshape(self.istarget, shape=(-1,))
        hit_at_k *= istarget
        ndcg_at_k = ndcg * istarget * hit_at_k

        return (tf.reduce_sum(hit_at_k), tf.reduce_sum(ndcg_at_k), tf.reduce_sum(istarget))

