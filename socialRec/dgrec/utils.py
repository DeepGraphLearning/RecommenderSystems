#coding=utf-8
from __future__ import print_function

import numpy as np
import pandas as pd
import random

def load_adj(data_path):
    df_adj = pd.read_csv(data_path + '/adj.tsv', sep='\t', dtype={0:np.int32, 1:np.int32})
    return df_adj    

def load_latest_session(data_path):
    ret = []
    for line in open(data_path + '/latest_sessions.txt'):
        chunks = line.strip().split(',')
        ret.append(chunks)
    return ret

def load_map(data_path, name='user'):
    if name == 'user':
        file_path = data_path + '/user_id_map.tsv'
    elif name == 'item':
        file_path = data_path + '/item_id_map.tsv'
    else:
        raise NotImplementedError
    id_map = {}
    for line in open(file_path):
        k, v = line.strip().split('\t')
        id_map[k] = str(v)
    return id_map

def load_data(data_path):
    adj = load_adj(data_path)
    latest_sessions = load_latest_session(data_path)
    user_id_map = load_map(data_path, 'user')
    item_id_map = load_map(data_path, 'item')
    train = pd.read_csv(data_path + '/train.tsv', sep='\t', dtype={0:np.int32, 1:np.int32, 3:np.float32})
    valid = pd.read_csv(data_path + '/valid.tsv', sep='\t', dtype={0:np.int32, 1:np.int32, 3:np.float32})
    test = pd.read_csv(data_path + '/test.tsv', sep='\t', dtype={0:np.int32, 1:np.int32, 3:np.float32})
    return [adj, latest_sessions, user_id_map, item_id_map, train, valid, test]


if __name__ == '__main__':
    # test data loading.
    data_path = 'path/to/data/'
    data = load_data(data_path)
