import numpy as np


class Evaluation:
    '''
    In progress...
    Eventually, we aim to include popular evaluation metrics as many as possible.
    '''
    def __init__(self, ks = [1, 5, 10, 20], ndcg_cutoff = 20):
        self.k = ks
        self.ndcg_cutoff = ndcg_cutoff
        self.clear()

    def clear(self):
        self.P = np.zeros_like(self.k, dtype=np.float32)
        self.R = np.zeros_like(self.k, dtype=np.float32)
        self.MAP = []
        self.NDCG = []

    def eval(self, user_id, target, prediction):
        '''
        :param user_id: int
        :param target: list of int
        :param prediction:  list of int
        :return:
        '''
        ranking = {}
        num_hits = 0.
        ap_score = 0.

        P = np.zeros_like(self.k, dtype=np.float32)
        for idx, item in enumerate(prediction):
            ranking[item] = idx + 1
            if item in target:
                for i, k in enumerate(self.k):
                    if idx < k:
                        P[i] += 1.0     # the predicted item is in top-k (Precise@K)
            if item in target and item not in prediction[:idx]:
                num_hits += 1.0
                ap_score += num_hits / (idx + 1.0)

        for i, k in enumerate(self.k):
            P[i] /= float(k)            # Precise@K should be divided by K
            
        self.P = self.P + P

        ap_score /= float(len(prediction))
        self.MAP.append(ap_score)

        R = np.zeros_like(self.k, dtype=np.float32)
        ndcg = 0
        for idx, item in enumerate(target):
            for i, k in enumerate(self.k):
                if item in prediction[:k]:
                    R[i] += 1           # the target is in top-k prediction (Recall@K)
            if ranking.get(item, 1e9) <= self.ndcg_cutoff:
                ndcg += 1.0 / np.log2(1.0 + ranking[item])
        ndcg /= float(len(target))
        self.NDCG.append(ndcg)

        R = R / float(len(target))      # Recall@K should be divided by number of targets
        self.R = self.R + R

    def result(self):
        num_data = len(self.MAP)
        self.P = self.P / float(num_data)
        self.R = self.R / float(num_data)
        print("==========================================")
        print("NDCG@%d = %8.4f" % (self.ndcg_cutoff, np.mean(self.NDCG)))
        print("MAP  = %8.4f" % np.mean(self.MAP))

        for i, k in enumerate(self.k):
            print("Precise @%2d = %6.4f" % (k, self.P[i]))

        for i, k in enumerate(self.k):
            print("Recall @%2d  = %6.4f" % (k, self.R[i]))
        print("==========================================")
