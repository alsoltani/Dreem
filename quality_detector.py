"""On this file you will implement you quality predictor function."""
from easy_import import extract_signals
import numpy as np
import pandas as pd
import pywt
import h5py
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, make_scorer, roc_auc_score
from sklearn.grid_search import GridSearchCV


class DictT(object):

    """Discrete Wavelet Transform operator."""

    def __init__(self, name, level):
        self.name = name
        self.level = level
        self.sizes = []

    def dot(self, mat):
        m = []

        if mat.shape[0] != mat.size:
            for i in xrange(mat.shape[1]):
                c = pywt.wavedec(mat[:, i], self.name, level=self.level)
                self.sizes.append(map(len, c))
                c = np.concatenate(c)
                m.append(c)
            return np.asarray(m).T
        else:
            c = pywt.wavedec(mat, self.name, level=self.level)
            self.sizes.append(map(len, c))
            return np.concatenate(c)


class Dict(object):

    """Inverse Wavelet Transform operator."""

    def __init__(self, name=None, sizes=None):
        self.name = name
        self.sizes = sizes
        assert name, sizes is not None

    def dot(self, m):
        d = []

        if m.shape[0] != m.size:
            for i in xrange(m.shape[1]):
                sizes_col = self.sizes[i]
                sizes = np.zeros(len(sizes_col) + 1)
                sizes[1:] = np.cumsum(sizes_col)
                c = [m[:, i][sizes[k]:sizes[k + 1]] for k in xrange(0, len(sizes) - 1)]
                d.append(pywt.waverec(c, self.name))
            return np.asarray(d).T

        else:
            sizes_col = self.sizes[0]
            sizes = np.zeros(len(sizes_col) + 1)
            sizes[1:] = np.cumsum(sizes_col)

            m = [m[sizes[k]:sizes[k + 1]] for k in xrange(0, len(sizes) - 1)]
            return pywt.waverec(m, self.name)


def train_model(quality_dataset="quality_dataset.h5", grid_search=False):

    """Trains the model on quality dataset."""

    # Data loading.

    hf = h5py.File(quality_dataset, 'r')
    quality_frame = pd.DataFrame(np.array(hf["dataset"]), columns=np.array(hf["feature_descriptipn"]))
    quality, quality_target = quality_frame.iloc[:, :-1], quality_frame.iloc[:, -1].astype(np.int32)
    raw_col, filtered_col = quality.columns[:500], quality.columns[500:]

    del quality_frame
    hf.close()

    # Discrete wavelet decomposition.

    wavelet_operator_t = DictT(level=None, name="db20")

    basis_t = wavelet_operator_t.dot(np.identity(len(raw_col)))
    basis_t /= np.sqrt(np.sum(basis_t ** 2, axis=0))

    # Test, val split.

    X_train, X_val, y_train, y_val = train_test_split(quality[filtered_col].dot(basis_t.T), quality_target,
                                                      test_size=0.2, random_state=42)

    precision_scorer = make_scorer(precision_score)

    # Grid-search parameter selection.

    if grid_search:
        clf = GridSearchCV(RandomForestClassifier(),
                           param_grid={
                               "n_estimators": [10, 50, 100],
                               "max_features": ["auto", "sqrt", "log2", None],
                               "max_depth": [5, 10, 20, None],
                               "min_samples_split": [1, 2, 5],
                               "min_samples_leaf": [1, 2, 5],
                               "bootstrap": [True, False],
                               "n_jobs": [-1]
                           },
                           scoring=precision_scorer)
    else:
        clf = RandomForestClassifier(n_estimators=10, max_features=None, max_depth=None)

    # Training.

    clf.fit(X_train, y_train)
    return clf, X_train.shape[1]


def window_stack(a, stepsize=1, width=4):

    temp = np.vstack(a[stepsize * i: stepsize * i + width] for i in range((len(a) - width) // stepsize + 1))
    return np.vstack((temp, a[-width:]))


def predict_quality(record, stepsize = 20):
    """Predict the quality of the signal.

    Input
    =====
    record: path to a record

    Output
    ======
    results: a list of 4 signals between 0 and 1 estimating the quality
    of the 4 channels of the record at each timestep.
    This results must have the same size as the channels.
    """

    # Extract signals from record
    raws, filtered = extract_signals(record)
    # Initialize results (same size as the channels)
    results = np.zeros((raws.shape))

    clf, n_col = train_model()
    width = n_col

    wavelet_operator_t = DictT(level=None, name="db20")
    basis_t = wavelet_operator_t.dot(np.identity(n_col))
    basis_t /= np.sqrt(np.sum(basis_t ** 2, axis=0))

    for i in xrange(len(filtered)):
        window_stack_array = window_stack(filtered[i], stepsize, width)
        res = clf.predict(window_stack_array.dot(basis_t.T))
        results[i] = np.concatenate((
            np.tile(np.atleast_2d(res[:-1]).T, (1, stepsize)).flatten(),
            np.tile(res[-1], (filtered[i].size - res[:-1].size * stepsize, 1)).flatten()
        ))

    return results


if __name__ == "__main__":
    record = "record1.h5"
    results = predict_quality(record)
