#coding: utf-8
'''
Author: Weiping Song
Contact: songweiping@pku.edu.cn
'''

import tensorflow as tf
import argparse
import numpy as np
import sys
import time
import math

from .utils import *
from .model import *
from .eval import Evaluation

parser = argparse.ArgumentParser(description='Sequential or session-based recommendation')
parser.add_argument('--model', type=str, default='tcn', help='sequential model: rnn/tcn/transformer. (default: tcn)')
parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--seq_len', type=int, default=20, help='max sequence length (default: 20)')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout (default: 0.2)')
parser.add_argument('--l2_reg', type=float, default=0.0, help='regularization scale (default: 0.0)')
parser.add_argument('--clip', type=float, default=1., help='gradient clip (default: 1.)')
parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for Adam (default: 0.001)')
parser.add_argument('--emsize', type=int, default=100, help='dimension of item embedding (default: 100)')
parser.add_argument('--neg_size', type=int, default=1, help='size of negative samples (default: 1)')
parser.add_argument('--worker', type=int, default=10, help='number of sampling workers (default: 10)')
parser.add_argument('--nhid', type=int, default=100, help='number of hidden units (default: 100)')
parser.add_argument('--levels', type=int, default=3, help='# of levels (default: 3)')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
parser.add_argument('--loss', type=str, default='ns', help='type of loss: ns/sampled_sm/full_sm (default: ns)')
parser.add_argument('--data', type=str, default='gowalla', help='data set name (default: gowalla)')
parser.add_argument('--log_interval', type=int, default=1e2, help='log interval (default: 1e2)')
parser.add_argument('--eval_interval', type=int, default=1e3, help='eval/test interval (default: 1e3)')

# ****************************** unique arguments for rnn model. *******************************************************
# None

# ***************************** unique arguemnts for tcn model.
parser.add_argument('--ksize', type=int, default=3, help='kernel size (default: 100)')

# ****************************** unique arguments for transformer model. *************************************************
parser.add_argument('--num_blocks', type=int, default=3, help='num_blocks')
parser.add_argument('--num_heads', type=int, default=2, help='num_heads')
parser.add_argument('--pos_fixed', type=int, default=0, help='trainable positional embedding usually has better performance')

args = parser.parse_args()
tf.set_random_seed(args.seed)

train_data, val_data, test_data, n_items, n_users = data_generator(args)

max_test_len = 20
test_data_per_step = prepare_eval_test(test_data, batch_size=100, max_test_len=max_test_len)

checkpoint_dir = '_'.join(['save', args.data, args.model, str(args.lr), str(args.l2_reg), str(args.emsize), str(args.dropout)])

print(args)
print ('#Item: ', n_items)
print ('#User: ', n_users)

model = NeuralSeqRecommender(args, n_items, n_users)

lr = args.lr

def evaluate_subsequent(source, sess):
    EV = Evaluation()
    for u in source.keys():
        itemids = source[u]
        uid = int(u)
        l = min(len(itemids), max_test_len)
        if l < 2:
            continue
        feed_dict = {model.inp:[itemids[:l-1]], model.dropout: 0}
        prediction = sess.run(model.prediction, feed_dict=feed_dict)
        prediction = prediction.flatten()
        for i in range(1, l):
            i_pred = prediction[(i-1)*(n_items+1): i*(n_items+1)]
            rank = np.argsort(-i_mi[1:]) + 1
            EV.eval(int(u), itemids[i:l], rank[:20])
    EV.result()

def evaluate(source, sess):
    total_hit_k = 0.0
    total_ndcg_k = 0.0
    count = 0.0
    for batch in source:
        feed_dict = {model.inp: batch[1], model.dropout: 0.}
        feed_dict[model.pos] = batch[2]
        hit, ndcg, n_target = sess.run([model.hit_at_k, model.ndcg_at_k, model.num_target], feed_dict=feed_dict)
        count += n_target
        total_hit_k += hit
        total_ndcg_k += ndcg

    val_hit = total_hit_k / count 
    val_ndcg = total_ndcg_k / count

    return [val_hit, val_ndcg]

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        model.saver.restore(sess, '{}/{}'.format(checkpoint_dir, 'model.ckpt'))
        print('Restore model successfully')
    else:
        print('Restore model failed!!!!!')

    test_hit, test_ndcg = evaluate(test_data_per_step, sess)
    print('Step-wise test :\nRecall@20 {:8.5f} | Ndcg@20 {:8.5f}'.format(test_hit, test_ndcg))
    #print('Subsequent as targets:\n')
    #evaluate_subsequent(test_data, sess)

if __name__ == '__main__':
    if not os.path.exists(checkpoint_dir):
        print('Checkpoint directory not found!')
        exit(0)
    main()
