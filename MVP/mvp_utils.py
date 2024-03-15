import itertools
from collections import OrderedDict
import numpy as np
# from keras import backend as K
import tensorflow as tf
from tensorflow.keras import backend as K
import networkx as nx

from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from matplotlib import pyplot as plt


def view_generator(view_gen, n_view, F):
    P = dict()
    
    # view generation
    if view_gen == 'R':
        # random view
        col_view = [i for i in range(F)]
        np.random.shuffle(col_view)
        partition_tmp = [i for i in range(1, F - 1)]
        np.random.shuffle(partition_tmp)
        partition = np.random.choice(partition_tmp, n_view - 1, replace=False)
        partition.sort()
        P['col_view'] = [np.sort(t).tolist() for t in np.split(col_view, partition)]
    elif view_gen == 'S':
        # intrinsic view
        P['viewpoints'] = 4
        P['col_view'] = [list(range(17)), [17], list(range(18, 79)), [79, 80, 81]]
    elif view_gen == 'E':
        # test for ensemble
        P['viewpoints'] = 4
        P['col_view'] = [list(range(F)), list(range(F)), list(range(F)), list(range(F))]
    elif view_gen == 'N':
        # no view
        P['viewpoints'] = 1
        P['col_view'] = [list(range(F))]
    else:
        raise ValueError('option should be S, R, E or N; got {}'.format(view_gen))
        
    return P


# for mini-batch Training
def create_batch(A_list, X_list):
    A_out = sp.block_diag(list(A_list))
    X_out = np.vstack(X_list)
    n_nodes = np.array([np.shape(a_)[0] for a_ in A_list])
    I_out = np.repeat(np.arange(len(n_nodes)), n_nodes)
    return A_out, X_out, I_out


def split_view(X, col_view):
    X_view_list = []
    for i, view in enumerate(col_view):
        tmp_view = []
        for x in X:
            x = np.array(x)
            tmp_view.append(x[:, view])
        X_view_list.append(np.array(tmp_view))
    return X_view_list


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    if len(keys) > 0:
        for instance in itertools.product(*vals):
            yield OrderedDict(zip(keys, instance))
    else:
        for _ in [dict(), ]:
            yield _


def plot_result(history, *ylims, **plots):
    '''
    @param history: [dict] corresponding values for each epoch
    @param ylim: [tuple] input for plt.ylim
    @param plots: [list] list for keys of history to show in a single plot

    'ylim' can be omitted. Once stated, number of ylim and plots should be the same.
    You can also give 'False' not to apply ply.ylim

    'plots' should have the following format
        title=['key1','key2',...]
    The 'title' will become a input to plt.title and 'key1',... will become a input to the key of history

    eg1. plot_result(history, acc=['train_acc','val_acc'], loss=['train_loss','val_loss'])
    eg2. plot_result(history, (30,100), False, train_accuracy=['train_acc'], losses=['train_loss','val_loss'])
    '''

    assert len(ylims) == 0 or len(ylims) == len(
        plots), 'number of ylims should be 0 or as same as number of plots, {} vs {}'.format(len(ylims), len(plots))

    iter = len(plots)
    titles = list(plots.keys())
    for i in range(iter):
        for dkey in plots[titles[i]]:
            plt.plot(history[dkey])
        plt.title(titles[i])
        plt.xlabel('epoch')
        if 'loss' in titles[i]:
            plt.ylabel('loss')
        else:
            plt.ylabel('accuracy')
        if len(ylims) > 0 and ylims[i] != False:
            plt.ylim(ylims[i][0], ylims[i][1])
        plt.legend(plots[titles[i]])
        plt.show()


def plot_histogram(rwidth=0.8, *xlim, **plots):
    '''
    xlim: [tuple] odd: plot number, even: xlim tuple. (eg. (2, (0,1), 4, (0,2), ...)
    plots: [dict] key: ['node', 'graph'] indicates the score is for each {key}
                  value: list of centrality score of length 3 ([0]: train data, [1]: val data, [2]: test data)
                         if value is empty list, skip plotting the corresponding graph
    '''

    order = list(xlim[::2])
    lim = list(xlim[1::2])
    assert len(order) == len(lim), 'number of plot number and xlim, {} vs {}'.format(len(order), len(lim))
    for i in range(len(order)):
        assert isinstance(order[i], int), 'plot number type should be int, got {}'.format(type(order[i]))
        assert isinstance(lim[i], list), 'xlim type should be list, got {}'.format(type(lim[i]))

    colorset = ['blue', 'green', 'orange']
    plotno = 0
    for key in plots.keys():
        assert key in ['node', 'graph'], 'the plot data should be [node, graph], got {}'.format(key)
        assert len(plots[key]) <= 3, 'the plot for each {} should be maximum 3, got {}'.format(key, len(plots[key]))

        for idx, centrality in enumerate(plots[key]):
            if len(centrality) == 0:
                continue
            plt.hist(centrality, rwidth=rwidth, color=colorset[idx % 3])
            plt.title('Betweeness-centrality of Pruned Nodes for each {} ({} data)'.format(key,
                                                                                           ['train', 'val', 'test'][
                                                                                               idx]))
            plt.xlabel('centrality value')
            if plotno in order:
                plt.xlim(lim[order.index(plotno)])
            plt.show()
            plotno += 1


def compute_centrality(adjacencies, indicators):
    assert len(adjacencies) == len(indicators), 'input size mismatch (#A != #I)'

    centrality_for_nodes = []
    centrality_for_graph = []

    num_graph = len(adjacencies)
    for i in range(num_graph):
        A, I = adjacencies[i].numpy(), indicators[i].numpy().tolist()
        G = nx.from_numpy_matrix(A)
        centrality = nx.betweenness_centrality(G)
        pruned_nodes = np.where(np.array(I) == 0.0)[0].tolist()
        if len(pruned_nodes) > 0:
            centrality_sum = sum([centrality[node] for node in pruned_nodes])
            centrality_for_nodes.append(centrality_sum)
            centrality_for_graph.append(centrality_sum / len(pruned_nodes))
        else:
            centrality_for_graph.append(-1)

    return centrality_for_nodes, centrality_for_graph


def plot_pruned_graph(adjacencies, indicators, mode='all'):
    assert len(adjacencies) == len(indicators), 'input size mismatch (#A != #I)'

    num_graph = len(adjacencies)

    if mode == 'all':
        plot_list = list(range(num_graph))
    elif 'random' in mode:
        try:
            plot_list = np.random.choice(np.array(range(num_graph)), int(mode[6:])).tolist()
        except:
            raise ValueError('invalid mode')
    elif 'head' in mode:
        try:
            plot_list = list(range(num_graph))[:int(mode[4:])]
        except:
            raise ValueError('invalid mode')
    else:
        raise ValueError('invalid mode')

    for i in range(num_graph):
        if not i in plot_list:
            continue
        A, I = adjacencies[i].numpy(), indicators[i].numpy()
        G = nx.from_numpy_matrix(A)
        color = np.array(['black'] * len(I))
        color[np.where(I == 0.0)[0].tolist()] = 'red'
        nx.draw(G, node_color=color.tolist())
        plt.show()
