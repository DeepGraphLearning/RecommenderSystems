#coding=utf-8
'''
Author: Weiping Song
Contact: songweiping@pku.edu.cn
Reference: https://github.com/kang205/SASRec/blob/master/sampler.py
'''

# Disclaimer:
# Part of this file is derived from 
# https://github.com/kang205/SASRec/

import numpy as np
from multiprocessing import Process, Queue

def random_neg(pos, n, s):
    '''
    p: positive one
    n: number of items
    s: size of samples.
    '''
    neg = set()
    for _ in range(s):
        t = np.random.randint(1, n+1)
        while t in pos or t in neg:
            t = np.random.randint(1, n+1)
        neg.add(t)
    return list(neg)

def sample_function(data, n_items, n_users, batch_size, max_len, neg_size, result_queue, SEED, neg_method='rand'):
    '''
    data: list of train data, key: user, value: a set of all user's clicks.
    tensors: list of train tensors, each element of list is also a list.
    masks: list of train masks, each element of list is also a list.
    batch_size: number of samples in a batch.
    neg_size: number of negative samples.
    '''
    num_samples = np.array([len(data[str(u)]) for u in range(1, n_users+1)])
    prob_ = num_samples / (1.0 * np.sum(num_samples))
    def sample():
        # sample a user based on behavior frequency.
        user = np.random.choice(a=range(1,1+n_users), p=prob_)
        u = str(user)

        # sample a slice from user u randomly. 
        if len(data[u]) <= max_len:
            idx = 0
        else:
            idx = np.random.randint(0, len(data[u])-max_len+1)
        seq = np.zeros([max_len], dtype=np.int32)
        for i, itemid in enumerate(data[u][idx:idx+max_len]):
            seq[i] = itemid

        pos = np.zeros([max_len], dtype=np.int32)
        neg = np.zeros([max_len, neg_size], dtype=np.int32)
        l = len(data[u]) - idx - 1
        l = min(l, max_len)
        for j in range(l):
            pos[j] = data[u][idx+1+j]
            if neg_method == 'rand':
                neg[j,:] = random_neg([pos[j]], n_items, neg_size)
            else: # Currently we only support random negative samples.
                raise NotImplementedError
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        result_queue.put(list(zip(*one_batch)))

class Sampler(object):
    def __init__(self, data, n_items, n_users, batch_size=128, max_len=20, neg_size=10, n_workers=10, neg_method='rand'):
        self.result_queue = Queue(maxsize=int(2e5))
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(data,
                                                    n_items, 
                                                    n_users,
                                                    batch_size, 
                                                    max_len, 
                                                    neg_size, 
                                                    self.result_queue, 
                                                    np.random.randint(2e9),
                                                    neg_method)))
            self.processors[-1].daemon = True
            self.processors[-1].start()
    
    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
