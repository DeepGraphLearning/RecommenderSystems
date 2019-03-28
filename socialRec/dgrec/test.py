#coding=utf-8
from __future__ import division
from __future__ import print_function

import os, sys
import argparse
import tensorflow as tf
import numpy as np
import time

from .utils import *
from .minibatch import MinibatchIterator
from .model import DGRec



seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

def evaluate(sess, model, minibatch, val_or_test='val'):
    epoch_val_cost = []
    epoch_val_recall = []
    epoch_val_ndcg = []
    epoch_val_point = []
    input_str = []
    while not minibatch.end_val(val_or_test):
        feed_dict = minibatch.next_val_minibatch_feed_dict(val_or_test)
        x = np.reshape(feed_dict[minibatch.placeholders['input_x']], -1).tolist()
        x_str = '_'.join([str(v) for v in x if v !=0])
        input_str.append(x_str)
        outs = sess.run([model.loss,model.sum_recall, model.sum_ndcg, model.point_count], feed_dict=feed_dict)
        epoch_val_cost.append(outs[0])
        epoch_val_recall.append(outs[1])
        epoch_val_ndcg.append(outs[2])
        epoch_val_point.append(outs[3])
    return [np.mean(epoch_val_cost), np.sum(epoch_val_recall) / np.sum(epoch_val_point), np.sum(epoch_val_ndcg) / np.sum(epoch_val_point), epoch_val_recall, epoch_val_ndcg, input_str]

def construct_placeholders(args):
    # Define placeholders
    placeholders = {
        'input_x': tf.placeholder(tf.int32, shape=(args.batch_size, args.max_length), name='input_session'),
        'input_y': tf.placeholder(tf.int32, shape=(args.batch_size, args.max_length), name='output_session'),
        'mask_y': tf.placeholder(tf.float32, shape=(args.batch_size, args.max_length), name='mask_x'),
        'support_nodes_layer1': tf.placeholder(tf.int32, shape=(args.batch_size*args.samples_1*args.samples_2), name='support_nodes_layer1'),
        'support_nodes_layer2': tf.placeholder(tf.int32, shape=(args.batch_size*args.samples_2), name='support_nodes_layer2'),
        'support_sessions_layer1': tf.placeholder(tf.int32, shape=(args.batch_size*args.samples_1*args.samples_2,\
                                    args.max_length), name='support_sessions_layer1'),
        'support_sessions_layer2': tf.placeholder(tf.int32, shape=(args.batch_size*args.samples_2,\
                                    args.max_length), name='support_sessions_layer2'),
        'support_lengths_layer1': tf.placeholder(tf.int32, shape=(args.batch_size*args.samples_1*args.samples_2), 
                                    name='support_lengths_layer1'),
        'support_lengths_layer2': tf.placeholder(tf.int32, shape=(args.batch_size*args.samples_2), 
                                    name='support_lengths_layer2'),
    }
    return placeholders

def test(args, data):
    adj_info = data[0]
    latest_per_user_by_time = data[1]
    user_id_map = data[2]
    item_id_map = data[3]
    train_df = data[4]
    valid_df = data[5]
    test_df = data[6]
    
    args.num_items = len(item_id_map) + 1
    args.num_users = len(user_id_map)
    args.batch_size = 1
    placeholders = construct_placeholders(args)
    
    minibatch = MinibatchIterator(adj_info,
                latest_per_user_by_time,
                [train_df, valid_df, test_df],
                placeholders,
                batch_size=args.batch_size,
                max_degree=args.max_degree,
                num_nodes=len(user_id_map),
                max_length=args.max_length,
                samples_1_2=[args.samples_1, args.samples_2],
                training=False)
    
    dgrec = DGRec(args, minibatch.sizes, placeholders)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
    ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore model from {}!'.format(args.ckpt_dir))
    else:
        print('Failed to restore model from {}'.format(args.ckpt_dir))
        sys.exit(0)
    ret = evaluate(sess, dgrec, minibatch, "test")
    print("Test results(batch_size=1):",
          "\tloss=", "{:.5f}".format(ret[0]),
          "\trecall@20=", "{:.5f}".format(ret[1]),
          "\tndcg=", "{:.5f}".format(ret[2]),
          )

    recall = ret[-3]
    ndcg = ret[-2]
    x_strs = ret[-1]
    with open('metric_dist.txt','w') as f:
        for idx in range(len(ret[-1])):
            f.write(x_strs[idx] + '\t' + str(recall[idx]) + '\t' + str(ndcg[idx]) + '\n')

class Args():
    training = False
    global_only = False
    local_only = False
    epochs = 20
    aggregator_type='attn'
    act='linear'
    batch_size = 200
    max_degree = 50
    num_users = -1
    num_items = 100
    concat=False
    learning_rate=0.001
    hidden_size = 100
    embedding_size = 100
    emb_user = 100
    max_length=20
    samples_1=10
    samples_2=5
    dim1 = 100
    dim2 = 100
    model_size = 'small'
    dropout = 0.
    weight_decay = 0.
    print_every = 100
    val_every = 500
    ckpt_dir = 'save/'

def parseArgs():
    args = Args()
    parser = argparse.ArgumentParser(description='DGRec args')
    parser.add_argument('--batch', default=200, type=int)
    parser.add_argument('--model', default='attn', type=str)
    parser.add_argument('--act', default='relu', type=str)
    parser.add_argument('--degree', default=50, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--hidden', default=100, type=int)
    parser.add_argument('--embi', default=50, type=int)
    parser.add_argument('--embu', default=50, type=int)
    parser.add_argument('--samples1', default=10, type=int)
    parser.add_argument('--samples2', default=5, type=int)
    parser.add_argument('--dim1', default=100, type=int)
    parser.add_argument('--dim2', default=100, type=int)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--decay_steps', default=400, type=int)
    parser.add_argument('--decay_rate', default=0.98, type=float)
    parser.add_argument('--local', default=0, type=int)
    parser.add_argument('--glb', default=0, type=int)
    new_args = parser.parse_args()

    args.batch_size = new_args.batch
    args.aggregator_type= new_args.model
    args.act= new_args.act
    args.max_degree = new_args.degree
    args.learning_rate = new_args.lr
    args.hidden_size = new_args.hidden
    args.embedding_size = new_args.embi
    args.emb_user = new_args.embu
    args.samples_1 = new_args.samples1
    args.samples_2 = new_args.samples2
    args.dim1 = new_args.dim1
    args.dim2 = new_args.dim2
    args.dropout = new_args.dropout
    args.weight_decay = new_args.l2
    args.decay_steps = new_args.decay_steps
    args.decay_rate = new_args.decay_rate
    args.local_only = new_args.local
    args.global_only = new_args.glb
    args.ckpt_dir = args.ckpt_dir + 'dgrec_batch{}'.format(args.batch_size)
    args.ckpt_dir = args.ckpt_dir + '_model{}'.format(args.aggregator_type)
    args.ckpt_dir = args.ckpt_dir + '_act{}'.format(args.act)
    args.ckpt_dir = args.ckpt_dir + '_maxdegree{}'.format(args.max_degree)
    args.ckpt_dir = args.ckpt_dir + '_lr{}'.format(args.learning_rate)
    args.ckpt_dir = args.ckpt_dir + '_hidden{}'.format(args.hidden_size)
    args.ckpt_dir = args.ckpt_dir + '_embi{}'.format(args.embedding_size)
    args.ckpt_dir = args.ckpt_dir + '_embu{}'.format(args.emb_user)
    args.ckpt_dir = args.ckpt_dir + '_samples1st{}'.format(args.samples_1)
    args.ckpt_dir = args.ckpt_dir + '_samples2nd{}'.format(args.samples_2)
    args.ckpt_dir = args.ckpt_dir + '_dim1st{}'.format(args.dim1)
    args.ckpt_dir = args.ckpt_dir + '_dim2nd{}'.format(args.dim2)
    args.ckpt_dir = args.ckpt_dir + '_dropout{}'.format(args.dropout)
    args.ckpt_dir = args.ckpt_dir + '_l2reg{}'.format(args.weight_decay)
    args.ckpt_dir = args.ckpt_dir + '_decaysteps{}'.format(args.decay_steps)
    args.ckpt_dir = args.ckpt_dir + '_decayrate{}'.format(args.decay_rate)
    args.ckpt_dir = args.ckpt_dir + '_global{}'.format(new_args.glb)
    args.ckpt_dir = args.ckpt_dir + '_local{}'.format(new_args.local)
    return args

def main(argv=None):
    args = parseArgs()
    print('Loading data..')
    data = load_data('data/')
    print("Done loading data..")
    test(args, data)

if __name__ == '__main__':
    tf.app.run()
