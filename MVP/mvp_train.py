import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import sys
import json
import math
from collections import OrderedDict

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from spektral.utils import batch_iterator, log, init_logging

import numpy as np
import scipy.sparse as sp

from mvp_layers import *
from mvp_models import load_model
from mvp_utils import *
from mvp_parser import load_dataset


class Trainer:
    
    def __init__(self, args):
        self.P = dict()
        self.P.update(vars(args))
        

    def train_step(self, inputs, labels):
        """
        :param inputs: [X, A, I, X1, X2, ...]
        :param labels: y
        """
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = self.pool_loss(labels, prediction)
            loss += sum(self.model.losses)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.total_train_loss(loss)
        self.total_train_accuracy(labels, prediction)


    def test_step(self, inputs, labels):
        
        prediction = self.model(inputs)
        t_loss = self.pool_loss(labels, prediction)
        t_loss += sum(self.model.losses)

        self.total_test_loss(t_loss)
        self.total_test_accuracy(labels, prediction)


    def evaluate(self, batches_):
        
        for data in batches_:
            sX_test_view = []

            A_, X_, I_ = create_batch(data[1], data[0]) 
            A_ = tf.cast(tf.sparse.to_dense(
                tf.sparse.SparseTensor(
                    indices=np.array([A_.row, A_.col]).T,
                    values=A_.data,
                    dense_shape=A_.shape)
            ), dtype=tf.float32)

            X_test_tensor = tf.convert_to_tensor(X_, dtype=tf.float32)
            A_test_tensor = tf.convert_to_tensor(A_, dtype=tf.float32) 
            I_test_tensor = tf.convert_to_tensor(I_, dtype=tf.float32) 

            for i in range(self.P["viewpoints"]): 
                sX_test_view.append(tf.convert_to_tensor(data[i + 2][0], dtype=tf.float32)) 
            y_vl = tf.convert_to_tensor([data[self.P["viewpoints"] + 2][0]], dtype=tf.float32)

            self.test_step([X_test_tensor, A_test_tensor, I_test_tensor] + sX_test_view, y_vl)

        t_loss = self.total_test_loss.result()
        t_acc = self.total_test_accuracy.result()

        return t_loss, t_acc


    def train(self, pretrain=None):
        
        # pretraining config
        mvp_trainable = True
        pool_trainable = True
        if pretrain is None:
            pass
        elif pretrain == 'pre':
            mvp_trainable = False
            self.P['history_path'] += '_pre'
        elif pretrain == 'post_ox':
            pool_trainable = False
            self.P['history_path'] = self.P['history_path'][:-4] + '_post_ox'
        elif pretrain == 'post_oo':
            self.P['history_path'] = self.P['history_path'][:-4] + '_post_oo'
        else:
            raise ValueError("Invalid Argument '{}'".format(pretrain))
        
        # Load raw data
        if not os.path.isdir(os.path.join('./dataset', self.P['dataset'])):
            os.system('tar -xzvf {}.tar.gz'.format(self.P['dataset'][:-1]))
        X = np.load(os.path.join('./dataset', self.P['dataset'], 'X.npy'), allow_pickle=True)
        A = np.load(os.path.join('./dataset', self.P['dataset'], 'A.npy'), allow_pickle=True)
        y = np.load(os.path.join('./dataset', self.P['dataset'], 'y.npy'), allow_pickle=True)

        # Parameters
        F = np.shape(X[0])[-1] # Dimension of node features
        n_out = np.shape(y)[-1]  # Dimension of the target
        average_N = np.ceil(np.mean([np.array(a).shape[-1] for a in A]))  # Average number of nodes in dataset

        if pretrain is None or pretrain == 'pre':
            self.P.update(view_generator(self.P['view_gen'], self.P['viewpoints'], F))
        if isinstance(self.P['hidden_dim'], int):
            self.P['hidden_dim'] = [self.P['hidden_dim']] * self.P['viewpoints']

        os.makedirs(self.P['history_path'])
        log_dir = init_logging('../' + self.P['history_path'])  # Create log directory and file
        log(self.P)

        accuracy_list = []

        for run in range(self.P['runs']):

            # Load dataset
            dataset_train, dataset_val, dataset_test, seed = load_dataset(X, A, y, self.P)
            log('seed: {}'.format(seed))

            # Load model
            self.model = load_model(self.P, F, n_out, mvp_trainable, pool_trainable)
            if pretrain is not None and pretrain != 'pre':
                self.model.load_weights(os.path.join(self.P['history_path'][:-7]+'pre', 'best_model_{}.h5'.format(run)))
            self.model.summary()

            # model setting
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.P['learning_rate'])
            self.pool_loss = tf.keras.losses.categorical_crossentropy

            # evaluation metrics
            self.total_train_loss = tf.keras.metrics.Mean(name='total_train_loss')
            self.total_train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='total_train_accuracy')
            self.total_test_loss = tf.keras.metrics.Mean(name='total_test_loss')
            self.total_test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='total_test_accuracy')

            # Variable initialization
            best_val_loss = np.inf
            patience = self.P['es_patience']
            current_batch = 0
            batches_in_epoch = 1 + int(len(y)*0.81)
            total_batches = batches_in_epoch * self.P['epochs']
            start_time = time.time()
            epoch_time = [0]

            # Training
            log('Fitting model {} Runs'.format(run))
            batches = batch_iterator(dataset_train, batch_size=1, epochs=self.P['epochs'])

            for data in batches:
                sX_train_view = []
                A_, X_, I_ = create_batch(data[1], data[0])
                A_ = tf.cast(tf.sparse.to_dense(
                    tf.sparse.SparseTensor(
                        indices=np.array([A_.row, A_.col]).T,
                        values=A_.data,
                        dense_shape=A_.shape)
                ), dtype=tf.float32)

                X_train_tensor = tf.convert_to_tensor(X_, dtype=tf.float32)
                A_train_tensor = tf.convert_to_tensor(A_, dtype=tf.float32)
                I_train_tensor = I_ #1115

                for i in range(self.P["viewpoints"]):
                    sX_train_view.append(tf.convert_to_tensor(data[i + 2][0], dtype=tf.float32))

                epoch_time[-1] -= time.time()
                y_tr = tf.convert_to_tensor(np.array([data[self.P["viewpoints"] + 2][0]]), dtype=tf.float32)
                self.train_step([X_train_tensor, A_train_tensor, I_train_tensor] + sX_train_view, y_tr)
                epoch_time[-1] += time.time()

                current_batch += 1
                if current_batch % batches_in_epoch == 0:
                    batches_val = batch_iterator(dataset_val, batch_size=1, shuffle=True)
                    val_loss, val_acc = self.evaluate(batches_val)

                    ep = int(current_batch / batches_in_epoch)
                    log('Ep: {:d} - Loss: {:.2f} - Acc: {:.2f} - Val loss: {:.2f} - Val acc: {:.2f} - Average epoch time: {:.2f}'
                        .format(ep, self.total_train_loss.result(), self.total_train_accuracy.result() * 100, val_loss,
                                val_acc.numpy() * 100, np.mean(epoch_time)))
                    epoch_time.append(0)

                    self.total_test_loss.reset_states()
                    self.total_test_accuracy.reset_states()
                    self.total_train_loss.reset_states()
                    self.total_train_accuracy.reset_states()

                    if float(val_loss.numpy()) < best_val_loss:
                        best_val_loss = float(val_loss.numpy())
                        patience = self.P['es_patience']
                        log('New best val_loss {:.3f}'.format(best_val_loss))
                        self.model.save_weights(os.path.join(log_dir, 'best_model_{}.h5'.format(run)))
                    else:
                        patience -= 1
                        if patience == 0:
                            log('Early stopping (best val_loss: {})'.format(best_val_loss))
                            break
            epoch_time[-1] += time.time()


            # Test Model
            log('Testing model')

            self.model.load_weights(os.path.join(log_dir, 'best_model_{}.h5'.format(run)))
            batches_test = batch_iterator(dataset_test, batch_size=1, shuffle=True)

            test_loss, test_acc = self.evaluate(batches_test)
            log('Done.\nTest loss: {:.2f} - Test acc: {:.2f}'
                .format(test_loss.numpy(), test_acc.numpy() * 100))

            accuracy_list.append(test_acc.numpy())

            self.total_test_loss.reset_states()
            self.total_test_accuracy.reset_states()

        log('Overall acc performance: {} =- {}'.format(np.mean(accuracy_list), np.std(accuracy_list)))

