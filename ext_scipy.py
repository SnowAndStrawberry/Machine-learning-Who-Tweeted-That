# COMP90051 Project 1 source code
# For Team 192

"""
ext_scipy.py
Extends scipy's classes to allow handling large datasets by incrementally
training a model (using fit_partial) with its fragments.

-- For COMP90051 Project 1
-- Written by: Andy P. (696382)

"""

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from scipy.sparse import issparse

import time
from datetime import datetime

import numpy as np
import pandas as pd
from IPython.display import display, clear_output

# IMPROVEMENT:
#  We now train in incrementally, instead of an entire batch.
#  This avoids crashing memory issue.

class SparseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model, batch_amount=2000):
        self.model = model
        self.batch_amount = batch_amount
        
    def fit(self, X, y=None, **fit_params):
        time_start = time.time()
        i = 0
        
        while i * self.batch_amount < X.shape[0]:
            clear_output(wait=True)
            
            if y is None:
                y_use = None,
            else:
                y_use = y[self.batch_amount * i : self.batch_amount * (i+1)],
            
            display("Transform progress: %d/%d (%.4f)    [%.3f s]" 
                    % (i*self.batch_amount, X.shape[0], float(i*self.batch_amount)/X.shape[0], time.time() - time_start))
            self.model.partial_fit(X[self.batch_amount * i : self.batch_amount * (i+1)].toarray(),
                              y_use, **fit_params)
            i += 1
            
        return self
    
    def transform(self, X, **transform_params):
        return self.model.transform(X, **transform_params)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)
    
    def set_params(self, **params):
        return self.model.set_params(**params)


class IncrementalLearn(BaseEstimator, ClassifierMixin):
    """A classifier on top of a classifier compatible with learning incrementally."""
    
    def __init__(self, model, batch_amount=2000, convert_dense=False):
        self.model = model

        self.params_ = {}
        self.params_['batch_amount'] = batch_amount
        self.params_['convert_dense'] = convert_dense

        self.get_params = self.model.get_params
        self.set_params = self.model.set_params
        self.score = self.model.score
        self.predict = self.model.score
        self.partial_fit = self.model.partial_fit
    
    def fit(self, X, y, extra_text='', **fit_params):
        time_start = time.time()
        i = 0

        while i * self.params_['batch_amount'] < X.shape[0]:
            clear_output(wait=True)
            display(extra_text)
            display("Training progress: %d/%d (%.4f)" 
                    % (i*self.params_['batch_amount'], X.shape[0], float(i*self.params_['batch_amount'])/X.shape[0]))
            display("Time elapsed: %2.f secs" % (time.time() - time_start))
            
            idx_start = self.params_['batch_amount'] * i
            idx_end = min(X.shape[0], self.params_['batch_amount'] * (i+1))

            X_use = X[idx_start:idx_end]

            if issparse(X) and self.params_['convert_dense']:
                X_use = X_use.toarray() 
            
            self.model.partial_fit(X_use, y[idx_start:idx_end], **fit_params)
            i += 1

        return self



def predict_batch(X, f_predict, batch_amount=5000, sparse_expand=True):
    i = 0
    predictions = []
    start_time = time.time()
    
    while i * batch_amount < X.shape[0]:
        clear_output(wait=True)
        display("%d / %d (%.4f, %.3f s)" % (i * batch_amount, X.shape[0], (i * batch_amount)/X.shape[0], time.time() - start_time))
        
        if issparse(X) and sparse_expand:
            predictions.append(f_predict(X[i * batch_amount : (i+1)  * batch_amount].toarray()))
        else:
            predictions.append(f_predict(X[i * batch_amount : (i+1)  * batch_amount]))
            
        i += 1
    
    return np.concatenate(predictions)


def save_predictions(predictions, name=None):
    output = pd.DataFrame(
        data={'Predicted': predictions},
        index=range(1, len(predictions) + 1))
    output.index.name = 'Id'
    
    if name and type(name) == str:
        filename = name
    else:
        filename = 'output ' + datetime.now().strftime("%d-%m-%Y %H:%M:%S") + '.txt'
    output.to_csv(filename, sep=',')