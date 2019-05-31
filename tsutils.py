import numpy as np


class SequenceSpliter:

    """A class that tranforms a sequence into a supervised learning problem"""
    def __init__(self, lookback, look_ahead, step=1):
        self.lookback = lookback
        self.look_ahead = look_ahead
        self.step = step

    def fit(self, X):
        return X

    def transform(self, x):
        X, y = [], []
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        for i in range(0, len(x), self.step):
            # find the end of this pattern
            end_ix = i + self.lookback
            out_end_idx = end_ix + self.look_ahead
            # check if we are beyond the dataset
            if out_end_idx > len(x):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = x[i:end_ix, :], x[end_ix:out_end_idx, :]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def fit_transform(self, X):
        X = self.fit(X)
        x, y = self.transform(X)
        return x, y
