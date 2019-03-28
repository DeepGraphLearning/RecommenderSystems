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
        '''
        # sample a user based on behavior frequency.
        #TODO: more efficient non-uniform sampling method.
        Compute utility lists for non-uniform sampling from discrete distributions.
        Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        for details
        '''
        
        user = np.random.choice(a=range(1,1+n_users), p=prob_)
        u = str(user)

        # sample a slice from user u randomly. 
        idx = np.random.randint(1, len(data[u]))
        start = 0 if idx >= max_len else max_len - idx
        len_of_item = max_len - start

        # Assume max_len is set to 5, and we want to predict the 4-th entry in the sequence
        # Then the length of historical items is 3.
        # The following code will return the array like [0, 0, x, x, x]
        # i.e. the zero is padded to the left.
        seq = np.zeros([max_len], dtype=np.int32)
        seq[start:] = data[u][idx-len_of_item:idx]


        pos = data[u][idx]
        neg = np.zeros([neg_size], dtype=np.int32)


        if neg_method == 'rand':
            neg = random_neg([pos], n_items, neg_size)
        else:
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