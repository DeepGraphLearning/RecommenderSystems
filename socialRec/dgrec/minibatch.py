#coding=utf-8
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import sys
from .neigh_samplers import UniformNeighborSampler
from .utils import *

np.random.seed(123)

class MinibatchIterator(object):
    
    def __init__(self, 
                adj_info, # in pandas dataframe
                latest_sessions,
                data, # data list, either [train, valid] or [train, valid, test].
                placeholders,
                batch_size,
                max_degree,
                num_nodes,
                max_length=20,
                samples_1_2=[10,5],
                training=True):
        self.num_layers = 2 # Currently, only 2 layer is supported.
        self.adj_info = adj_info
        self.latest_sessions = latest_sessions
        self.training = training
        self.train_df, self.valid_df, self.test_df = data
        self.all_data = pd.concat(data)
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.num_nodes = num_nodes
        self.max_length = max_length
        self.samples_1_2 = samples_1_2
        self.sizes = [1, samples_1_2[1], samples_1_2[1]*samples_1_2[0]]
        self.visible_time = self.user_visible_time()
        self.test_adj, self.test_deg = self.construct_test_adj()
        if self.training:
            self.adj, self.deg = self.construct_adj()
            self.train_session_ids = self._remove_infoless(self.train_df, self.adj, self.deg)
            self.valid_session_ids = self._remove_infoless(self.valid_df, self.test_adj, self.test_deg)
            self.sampler = UniformNeighborSampler(self.adj, self.visible_time, self.deg)
        
        self.test_session_ids = self._remove_infoless(self.test_df, self.test_adj, self.test_deg)
       
        self.padded_data, self.mask = self._padding_sessions(self.all_data)
        self.test_sampler = UniformNeighborSampler(self.test_adj, self.visible_time, self.test_deg)
        
        self.batch_num = 0
        self.batch_num_val = 0
        self.batch_num_test = 0

    def user_visible_time(self):
        '''
            Find out when each user is 'visible' to her friends, i.e., every user's first click/watching time.
        '''
        visible_time = []
        for l in self.latest_sessions:
            timeid = max(loc for loc, val in enumerate(l) if val == 'NULL') + 1
            visible_time.append(timeid)
            assert timeid > 0 and timeid < len(l), 'Wrong when create visible time {}'.format(timeid)
        return visible_time

    def _remove_infoless(self, data, adj, deg):
        '''
        Remove users who have no sufficient friends.
        '''
        data = data.loc[deg[data['UserId']] != 0]
        reserved_session_ids = []
        print('sessions: {}\tratings: {}'.format(data.SessionId.nunique(), len(data)))
        for sessid in data.SessionId.unique():
            userid, timeid = sessid.split('_')
            userid, timeid = int(userid), int(timeid)
            cn_1 = 0
            for neighbor in adj[userid, : ]:
                if self.visible_time[neighbor] <= timeid and deg[neighbor] > 0:
                    cn_2 = 0
                    for second_neighbor in adj[neighbor, : ]:
                        if self.visible_time[second_neighbor] <= timeid:
                            break
                        cn_2 += 1
                    if cn_2 < self.max_degree:
                        break
                cn_1 += 1
            if cn_1 < self.max_degree:
                reserved_session_ids.append(sessid)
        return reserved_session_ids

    def _padding_sessions(self, data):
        '''
        Pad zeros at the end of each session to length self.max_length for batch training.
        '''
        data = data.sort_values(by=['TimeId']).groupby('SessionId')['ItemId'].apply(list).to_dict()
        new_data = {}
        data_mask = {}
        for k, v in data.items():
            mask = np.ones(self.max_length, dtype=np.float32)
            x = v[:-1]
            y = v[1: ]
            assert len(x) > 0
            padded_len = self.max_length - len(x)
            if padded_len > 0:
                x.extend([0] * padded_len)
                y.extend([0] * padded_len)
                mask[-padded_len: ] = 0.
            v.extend([0] * (self.max_length - len(v)))
            x = x[:self.max_length]
            y = y[:self.max_length]
            v = v[:self.max_length]
            new_data[k] = [np.array(x, dtype=np.int32), np.array(y, dtype=np.int32), np.array(v, dtype=np.int32)]
            data_mask[k] = np.array(mask, dtype=bool)
        return new_data, data_mask

    def _batch_feed_dict(self, current_batch):
        '''
        Construct batch inputs.
        '''
        current_batch_sess_ids, samples, support_sizes = current_batch
        feed_dict = {}
        input_x = []
        input_y = []
        mask_y = []
        timeids = []
        for sessid in current_batch_sess_ids:
            nodeid, timeid = sessid.split('_')
            timeids.append(int(timeid))
            x, y, _ = self.padded_data[sessid]
            mask = self.mask[sessid]
            input_x.append(x)
            input_y.append(y)
            mask_y.append(mask)
        feed_dict.update({self.placeholders['input_x']: input_x})
        feed_dict.update({self.placeholders['input_y']: input_y})
        feed_dict.update({self.placeholders['mask_y']: mask_y})

        feed_dict.update({self.placeholders['support_nodes_layer1']: samples[2]})
        feed_dict.update({self.placeholders['support_nodes_layer2']: samples[1]})
        #prepare sopportive user's recent sessions.
        support_layers_session = []
        support_layers_length = []
        for layer in range(self.num_layers):
            start = 0
            t = self.num_layers - layer
            support_sessions = []
            support_lengths = []
            for batch in range(self.batch_size):
                timeid = timeids[batch]
                support_nodes = samples[t][start: start + support_sizes[t]]
                for support_node in support_nodes:
                    support_session_id = str(self.latest_sessions[support_node][timeid])
                    support_session = self.padded_data[support_session_id][2]
                    #print(support_session)
                    length = np.count_nonzero(support_session)
                    support_sessions.append(support_session)
                    support_lengths.append(length)
                start += support_sizes[t]
            support_layers_session.append(support_sessions)
            support_layers_length.append(support_lengths)
        feed_dict.update({self.placeholders['support_sessions_layer1']:support_layers_session[0]})
        feed_dict.update({self.placeholders['support_sessions_layer2']:support_layers_session[1]})
        feed_dict.update({self.placeholders['support_lengths_layer1']:support_layers_length[0]})
        feed_dict.update({self.placeholders['support_lengths_layer2']:support_layers_length[1]})
        return feed_dict 

    def sample(self, nodeids, timeids, sampler):
        '''
        Sample neighbors recursively. First-order, then second-order, ...
        '''
        samples = [nodeids]
        support_size = 1
        support_sizes = [support_size]
        first_or_second = ['second', 'first']
        for k in range(self.num_layers):
            t = self.num_layers - k - 1
            node = sampler([samples[k], self.samples_1_2[t], timeids, first_or_second[t], support_size])
            support_size *= self.samples_1_2[t]
            samples.append(np.reshape(node, [support_size * self.batch_size,]))
            support_sizes.append(support_size)
        return samples, support_sizes

    def next_val_minibatch_feed_dict(self, val_or_test='val'):
        '''
        Construct evaluation or test inputs.
        '''
        if val_or_test == 'val':
            start = self.batch_num_val * self.batch_size
            self.batch_num_val += 1
            data = self.valid_session_ids
        elif val_or_test == 'test':
            start = self.batch_num_test * self.batch_size
            self.batch_num_test += 1
            data = self.test_session_ids
        else:
            raise NotImplementedError
        
        current_batch_sessions = data[start: start + self.batch_size]
        nodes = [int(sessionid.split('_')[0]) for sessionid in current_batch_sessions]
        timeids = [int(sessionid.split('_')[1]) for sessionid in current_batch_sessions]
        samples, support_sizes = self.sample(nodes, timeids, self.test_sampler)
        return self._batch_feed_dict([current_batch_sessions, samples, support_sizes])

    def next_train_minibatch_feed_dict(self):
        '''
        Generate next training batch data.
        '''
        start = self.batch_num * self.batch_size
        self.batch_num += 1
        current_batch_sessions = self.train_session_ids[start: start + self.batch_size]
        nodes = [int(sessionid.split('_')[0]) for sessionid in current_batch_sessions]
        timeids = [int(sessionid.split('_')[1]) for sessionid in current_batch_sessions]
        samples, support_sizes = self.sample(nodes, timeids, self.sampler)
        return self._batch_feed_dict([current_batch_sessions, samples, support_sizes])

    def construct_adj(self):
        '''
        Construct adj table used during training.
        '''
        adj = self.num_nodes*np.ones((self.num_nodes+1, self.max_degree), dtype=np.int32)
        deg = np.zeros((self.num_nodes,))
        missed = 0
        for nodeid in self.train_df.UserId.unique():
            neighbors = np.array([neighbor for neighbor in 
                                self.adj_info.loc[self.adj_info['Follower']==nodeid].Followee.unique()], dtype=np.int32)
            deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                missed += 1
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        #print('Unexpected missing during constructing adj list: {}'.format(missed))
        return adj, deg

    def construct_test_adj(self):
        '''
        Construct adj table used during evaluation or testing.
        '''
        adj = self.num_nodes*np.ones((self.num_nodes+1, self.max_degree), dtype=np.int32)
        deg = np.zeros((self.num_nodes,))
        missed = 0
        data = self.all_data
        for nodeid in data.UserId.unique():
            neighbors = np.array([neighbor for neighbor in 
                                self.adj_info.loc[self.adj_info['Follower']==nodeid].Followee.unique()], dtype=np.int32)
            deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                missed += 1
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        #print('Unexpected missing during constructing adj list: {}'.format(missed))
        return adj, deg

    def end(self):
        '''
        Indicate whether we finish a pass over all training samples.
        '''
        return self.batch_num * self.batch_size > len(self.train_session_ids) - self.batch_size
    
    def end_val(self, val_or_test='val'):
        '''
        Indicate whether we finish a pass over all testing or evaluation samples.
        '''
        batch_num = self.batch_num_val if val_or_test == 'val' else self.batch_num_test
        data = self.valid_session_ids if val_or_test == 'val' else self.test_session_ids
        end = batch_num * self.batch_size > len(data) - self.batch_size
        if end:
            if val_or_test == 'val':
                self.batch_num_val = 0
            elif val_or_test == 'test':
                self.batch_num_test = 0
            else:
                raise NotImplementedError
        if end:
            self.batch_num_val = 0
        return end

    def shuffle(self):
        '''
        Shuffle training data.
        '''
        self.train_session_ids = np.random.permutation(self.train_session_ids)
        self.batch_num = 0


if __name__ == '__main__':
    data = load_data('path/to/data')
    adj_info = data[0]
    latest_per_user_by_time = data[1]
    user_id_map = data[2]
    item_id_map = data[3]
    train_df = data[4]
    valid_df = data[5]
    test_df = data[6]
    minibatch = MinibatchIterator(adj_info,
                latest_per_user_by_time,
                [train_df, valid_df, test_df],
                None, #placeholders,
                batch_size=1,
                max_degree=50,
                num_nodes=len(user_id_map),
                max_length=20,
                samples_1_2=[10, 5])
    
