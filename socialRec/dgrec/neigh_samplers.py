from __future__ import division
from __future__ import print_function

import numpy as np

"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(object):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, visible_time, deg):
        self.adj_info = adj_info
        self.visible_time = visible_time
        self.deg = deg

    def __call__(self, inputs):
        nodeids, num_samples, timeids, first_or_second, support_size = inputs
        adj_lists = []
        for idx in range(len(nodeids)):
            node = nodeids[idx]
            timeid = timeids[idx // support_size]
            adj = self.adj_info[node, :]
            neighbors = []
            for neighbor in adj:
                if first_or_second == 'second':
                    if self.visible_time[neighbor] <= timeid:
                        neighbors.append(neighbor)
                elif first_or_second == 'first':
                    if self.visible_time[neighbor] <= timeid and self.deg[neighbor] > 0:
                        for second_neighbor in self.adj_info[neighbor]:
                            if self.visible_time[second_neighbor] <= timeid:
                                neighbors.append(neighbor)
                                break
            assert len(neighbors) > 0
            if len(neighbors) < num_samples:
                neighbors = np.random.choice(neighbors, num_samples, replace=True)
            elif len(neighbors) > num_samples:
                neighbors = np.random.choice(neighbors, num_samples, replace=False)
            adj_lists.append(neighbors)
        return np.array(adj_lists, dtype=np.int32)
