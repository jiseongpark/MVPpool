import numpy as np
import pandas as pd
import scipy.sparse as sp

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

from mvp_layers import *
from mvp_utils import *

def load_model(P, F, n_out, MVP_trainable=True, Pool_trainable=True):
    
    x_view = []  # input layer
    X_in = Input(shape=(F,), dtype=tf.float32, name='X_in')
    I_in = Input(shape=(), dtype=tf.int32, name='segment_ids_in')

    for i in range(P["viewpoints"]):
        x_view.append(tf.keras.Input(shape=(len(P['col_view'][i]),)))

    A_in = tf.keras.Input(shape=(None,))

    new_x_view = []
    for i in range(P["viewpoints"]):
        new_x_view.append(tf.keras.layers.Dense(P['embed_dim'], activation='relu')(x_view[i]))

    view = []  # GCN layer
    for i in range(P["viewpoints"]):
        view.append(GraphConvSkip(P['hidden_dim'][i],
                                  activation=P['activ'],
                                  kernel_regularizer=l2(P['GNN_l2']))([new_x_view[i], A_in]))

    if P['viewpoints'] == 1:
        z = view[0]
    else:
        z = tf.keras.layers.Concatenate(axis=1)(view)

    # reconstruction
    x_tilde = tf.keras.layers.Dense(F, activation=P['activ'])(z)
    a_tilde = tf.keras.activations.sigmoid(tf.linalg.matmul(z, tf.transpose(z)))
    pruning_input = [x_tilde, a_tilde, X_in, A_in]
    X_pruned, A_pruned = Pruning(F, P, ind_not_active=int(not MVP_trainable))(pruning_input)
    MVP = Model([X_in, A_in, I_in] + x_view, [X_pruned, A_pruned])


    # Model Build; Pooling
    if P['GNN_type'] == 'GCN':
        GNN = GraphConv
    elif P['GNN_type'] == 'ARMA':
        GNN = ARMAConv
    elif P['GNN_type'] == 'GCS' or P['GNN_type'] == 'GIN':
        GNN = GraphGIN
    else:
        raise ValueError('Unknown GNN type: {}'.format(P['GNN_type']))

    # Block 1
    if P['method'] == 'GMT':
        output = GraphMultisetTransformer(input_dim=F,
                                   hidden_dim=128,
                                   num_heads=2,
                                   p=0.5,
                                   avg_num_nodes= 284.32,  #PROTEIN
                                   pooling_ratio=0.25)([X_pruned, A_pruned])
    elif P['method'] == 'diff_pool':
        gc_in = X_pruned
        for _ in range(P['gnn_depth']):
            gc_in = GNN(P['n_channels'],
                        epsilon=P['epsilon'],
                        activation=P['activ'],
                        kernel_regularizer=l2(P['GNN_l2']))([gc_in, A_pruned])
        X_1, A_1, I_1, M_1 = DiffPool(k=P['K'][0],
                                      return_mask=True,
                                      channels=P['n_channels'],
                                      activation=P['activ'],
                                      kernel_regularizer=l2(P['GNN_l2']))([gc_in, A_pruned, I_in])
    elif P['method'] == 'dense':
        X_1 = Dense(P['n_channels'], activation=P['activ'], kernel_regularizer=l2(P['GNN_l2']))(X_pruned)
        A_1 = A_in
        I_1 = I_in
    else:
        gc1 = X_pruned
        for _ in range(P['gnn_depth']):
            gc1 = GNN(P['n_channels'],
                      epsilon=P['epsilon'],
                      activation=P['activ'],
                      kernel_regularizer=l2(P['GNN_l2']))([gc1, A_pruned])

        if P['method'] == 'top_k_pool':
            X_1, A_1, I_1, M_1 = TopKPool(0.8)([gc1, A_pruned, I_in])
        elif P['method'] == 'sag_pool':
            X_1, A_1, I_1 = SAGPool(0.8)([gc1, A_pruned, I_in])
        elif P['method'] == 'mincut_pool':
            X_1, A_1, I_1, M_1 = newMinCutPool(k=P['K'][0],
                                               temperature=P['temperature'][0],
                                            mlp_hidden=[P['mincut_H']],
                                            return_mask=True,
                                            activation=P['activ'],
                                            kernel_regularizer=l2(P['pool_l2']))([gc1, A_pruned, I_in])

        elif P['method'] == 'flat':
            X_1 = gc1
            A_1 = A_in
            I_1 = I_in
        else:
            raise ValueError

    # Block 2
    if P['method'] == 'GMT':
        pass
    elif P['method'] == 'diff_pool':
        for _ in range(P['gnn_depth']):
            X_1 = GNN(P['n_channels'],
                      epsilon=P['epsilon'],
                      activation=P['activ'],
                      kernel_regularizer=l2(P['GNN_l2']))([X_1, A_1])
        X_2, A_2, I_2, M_2 = DiffPool(k=P['K'][1],
                                      return_mask=True,
                                      channels=P['n_channels'],
                                      activation=P['activ'],
                                      kernel_regularizer=l2(P['GNN_l2']))([X_1, A_1, I_1])
    elif P['method'] == 'dense':
        X_2 = Dense(P['n_channels'], activation=P['activ'], kernel_regularizer=l2(P['GNN_l2']))(X_1)
        A_2 = A_1
        I_2 = I_1
    else:
        gc2 = X_1
        for _ in range(P['gnn_depth']):
            gc2 = GNN(P['n_channels'],
                      epsilon=P['epsilon'],
                      activation=P['activ'],
                      kernel_regularizer=l2(P['GNN_l2']))([gc2, A_1])
        if P['method'] == 'top_k_pool':
            X_2, A_2, I_2, M_2 = TopKPool(0.8)([gc2, A_1, I_1])
        elif P['method'] == 'sag_pool':
            X_2, A_2, I_2 = SAGPool(0.8)([gc2, A_1, I_1])
        elif P['method'] == 'mincut_pool':
            X_2, A_2, I_2, M_2 = newMinCutPool(k=P['K'][1],
                                               temperature=P['temperature'][1],
                                            mlp_hidden=[P['mincut_H']],
                                            activation=P['activ'],
                                            return_mask=True,
                                            kernel_regularizer=l2(P['pool_l2']))([gc2, A_1, I_1])
        elif P['method'] == 'flat':
            X_2 = gc2
            A_2 = A_1
            I_2 = I_1
        else:
            raise ValueError

    # Block 3
    if P['method'] == 'GMT':
        pass
    elif P['method'] == 'dense':
        X_3 = Dense(P['n_channels'], activation=P['activ'], kernel_regularizer=l2(P['GNN_l2']))(X_2)
    else:
        X_3 = GNN(P['n_channels'], activation=P['activ'], kernel_regularizer=l2(P['GNN_l2']))([X_2, A_2])

    # Output block
    if P['method'] == 'GMT':
        pass
    else:
        avgpool = GlobalAvgPool()([X_3, I_2])
        output = Dense(n_out, activation='softmax')(avgpool)

    model = Model(MVP.input, output)
    MVP.trainable = MVP_trainable
    if not Pool_trainable:
        for layer in model.layers[len(MVP.layers)-1:]:
            layer.trainable = False

    return model