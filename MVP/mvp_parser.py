import sys
import json
import math
from collections import OrderedDict

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import *
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import logging
tf.get_logger().setLevel(logging.ERROR)
from spektral.layers import GraphConv, GlobalAvgPool, ARMAConv, GraphConvSkip
from spektral.layers import MinCutPool, DiffPool, TopKPool, SAGPool
from spektral.utils import batch_iterator, log, init_logging
from spektral.utils.convolution import normalized_adjacency

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mvp_layers import *
from mvp_models import load_model
from mvp_utils import *

def load_dataset(X, A, y, P):
    
    seed = np.random.choice(100000)

    X_Folds, X_test, A_Folds, A_test, y_Folds, y_test = train_test_split(X, A, y, test_size=0.1,
                                                                                  stratify=y,
                                                                                  random_state=seed)
    
    X_train, X_val, A_train, A_val, y_train, y_val = train_test_split(X_Folds, A_Folds, y_Folds, test_size=0.1,
                                                                                  stratify=y_Folds,
                                                                                  random_state=seed)

    # Preprocessing adjacency matrices for convolution
    if P['GNN_type'] == 'GCS' or P['GNN_type'] == 'ARMA':
        A_train = np.array([normalized_adjacency(np.array(a)) for a in A_train])
        A_val = np.array([normalized_adjacency(np.array(a)) for a in A_val])
        A_test = np.array([normalized_adjacency(np.array(a)) for a in A_test])
    elif P['GNN_type'] == 'GCN':
        A_train = np.array([normalized_adjacency(np.array(a) + sp.eye(np.array(a).shape[0])) for a in A_train])
        A_val = np.array([normalized_adjacency(np.array(a) + sp.eye(np.array(a).shape[0])) for a in A_val])
        A_test = np.array([normalized_adjacency(np.array(a) + sp.eye(np.array(a).shape[0])) for a in A_test])
    elif P['GNN_type'] == 'GIN':
        A_train = np.array([np.array(a) for a in A_train])
        A_val = np.array([np.array(a) for a in A_val])
        A_test = np.array([np.array(a) for a in A_test])
    else:
        raise ValueError('Unknown GNN type: {}'.format(P['GNN_type']))
        
    X_train_view = split_view(X_train, P['col_view'])
    X_val_view = split_view(X_val, P['col_view'])
    X_test_view = split_view(X_test, P['col_view'])
    
    dataset_train = [X_train, A_train] + X_train_view + [y_train]
    dataset_val = [X_val, A_val] + X_val_view + [y_val]
    dataset_test = [X_test, A_test] + X_test_view + [y_test]

    return dataset_train, dataset_val, dataset_test, seed