import tensorflow as tf
from mvp_utils import *
from tensorflow.python.util import nest
from tensorflow.keras.layers import *

from spektral.layers.convolutional.graph_conv import GraphConv
from spektral.utils import normalized_adjacency

from tensorflow.keras import Sequential
from tensorflow.keras import activations, initializers, regularizers, constraints, backend as K
from spektral.layers import ops
from tensorflow.python.keras.utils import tf_utils
import math
from tensorflow.keras.models import Model

def get_shapes(data):
    shapes = None
    if all(hasattr(x, 'shape') for x in nest.flatten(data)):
        shapes = nest.map_structure(lambda x: x.shape, data)
    return shapes

class Div2(Layer):
    def __init__(self, temperature):
        super(Div2, self).__init__()
        self.t = float(temperature)

    def call(self, inputs):
        return inputs/self.t

    
class GraphMultisetTransformer(Model):
    def __init__(self,
                 input_dim=82, #num_feature
                 hidden_dim=128, #hidden_size
                 pool_input_dim=None,
                 activation=None,
                 num_heads=2,
                 n_out=2,
                 in_features=128, # classifier in_features
                 use_bias=True,
                 p=0.5, #dropout_ratio
                 avg_num_nodes=39.06, #avg num node of PROTEIN
                 pooling_ratio=0.25,
                 ln=True,
                 cluster=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphMultisetTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.in_features = hidden_dim
        self.use_bias = use_bias
        self.p = p
        self.pool_input_dim = pool_input_dim
        self.n_out=n_out
        self.num_heads = num_heads
        self.pooling_ratio = pooling_ratio
        self.avg_num_nodes = avg_num_nodes
        self.ln = ln
        self.cluster = cluster
        self.convs = self.get_convs()
        self.pools = self.get_pools()
        self.classifier = self.get_classifier()

        
    def call(self, inputs):
        x, a= inputs
        batch=tf.zeros(tf.shape(x)[0], dtype=tf.int32)
        xs = []

        for _ in range(3):
            x = tf.nn.relu(self.convs[_]([x, a])) # 원래 a아니고 edge_inde
            xs.append(x)
        x = tf.concat(xs, 1)
        
        for i, poollayer in enumerate(self.pools):
            if i == 0:
                batch_x, mask = tf.expand_dims(x, axis=0), tf.ones((1, tf.shape(x)[0]), dtype=tf.bool)
                extended_attention_mask = tf.expand_dims(mask,axis=1)
                extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)

                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                batch_x = poollayer(batch_x, attention_mask=extended_attention_mask, graph=(x, a, batch))

            else:
                batch_x = poollayer(batch_x, attention_mask=extended_attention_mask)
            extended_attention_mask = None

        batch_x = Dense(self.input_dim, activation='linear')(batch_x)
        batch_x = tf.squeeze(batch_x, axis=1)
        
        for _ in range(len(self.classifier)):
            batch_x = self.classifier[_](batch_x)
        
        return batch_x

            
    
    def get_convs(self):
        
        return [GraphConv(self.hidden_dim, name='conv1'),
                GraphConv(self.hidden_dim, name='conv2'),
                GraphConv(self.hidden_dim, name='conv3')]
    
    def get_pools(self):
        
        self.pool_input_dim = self.hidden_dim * 3 if self.pool_input_dim is None else self.pool_input_dim
        self.num_nodes = math.ceil(self.pooling_ratio * self.avg_num_nodes)
        
        return [PMA(self.pool_input_dim, self.num_heads, num_seeds=self.num_nodes, ln=self.ln, cluster=self.cluster, mab_conv='GCN', var_name='pma1111'),
                SAB(self.pool_input_dim, self.hidden_dim, self.num_heads, ln=self.ln, cluster=self.cluster),
                PMA(self.pool_input_dim, self.num_heads, num_seeds=1, ln=self.ln, cluster=self.cluster, mab_conv=None, var_name='pma2222')]
    
    
    def get_classifier(self):
        
        return [Dense(self.in_features, activation='linear',name='class1'),
                ReLU(name='class2'),
                Dropout(self.p,name='class3'),
                Dense(self.in_features//2, activation='linear',name='class4'),
                ReLU(name='class5'),
                Dropout(self.p,name='class6'),
                Dense(self.n_out, activation='softmax',name='class7')]
    
    
class PMA(Layer):
    def __init__(self, dim, num_heads, num_seeds, ln=False, cluster=False, mab_conv=None, var_name='None', **kwargs):
        super(PMA, self).__init__(**kwargs)
        self.S = tf.Variable(tf.initializers.GlorotUniform()(shape=(1, num_seeds, dim)),name=var_name)

        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, cluster=cluster, conv=mab_conv)
        
    def call(self, X, attention_mask=None, graph=None, return_attn=False):
        s = self.S
#         for _ in range(np.shape(X)[0]-1):
#             s = tf.concat([s,self.S],0)
        return self.mab(s, X, attention_mask, graph, return_attn)
    
    
class SAB(Layer):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, cluster=False, mab_conv=None, **kwargs):
        super(SAB, self).__init__(**kwargs)
        
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, cluster=cluster)
        
    def call(self, X, attention_mask=None, graph=None):
        return self.mab(X, X, attention_mask, graph)
        
        
class MAB(Layer):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, cluster=False, conv=None, **kwargs):
        super(MAB, self).__init__(**kwargs)
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = Dense(dim_V, activation='linear',name='mab1')
        
        self.fc_k, self.fc_v = self.get_fc_kv(dim_K, dim_V, conv)
        
        if ln:
            self.ln0 = LayerNormalization(axis=-1,name='mab2')
            self.ln1 = LayerNormalization(axis=-1,name='mab3')
        self.fc_o = Dense(dim_V, activation='linear',name='mab4')
        
        self.softmax_dim = 2
        if cluster == True:
            self.softmax_dim = 1
    
    def call(self, Q, K, attention_mask=None, graph=None, return_attn=False):
        Q = self.fc_q(Q)

        
        if graph is not None:
            x, a, I = graph
#             K = GraphConv(dim_V)(x, a) 
#             V = GraphConv(dim_V)(x, a) 
            K, V = self.fc_k([x, a]), self.fc_v([x, a])
    
            K = tf.expand_dims(K, axis=0)
            V = tf.expand_dims(V, axis=0)
        else:
            K, V = self.fc_k(K), self.fc_v(K)
            
        
        dim_split = self.dim_V // self.num_heads
        Q_ = tf.concat(tf.split(Q, math.ceil(np.shape(Q)[2]/dim_split), axis=2),0) # 1920 * 2 (MH map (Q))
        K_ = tf.concat(tf.split(K, math.ceil(np.shape(Q)[2]/dim_split), axis=2),0)
        V_ = tf.concat(tf.split(V, math.ceil(np.shape(Q)[2]/dim_split), axis=2),0)

        if attention_mask is not None:
            attention_mask = tf.concat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = tf.matmul(Q_, tf.transpose(K_, perm=[0, 2, 1])/math.sqrt(self.dim_V))
            A = tf.nn.softmax(attention_mask + attention_score, self.softmax_dim)
        else:
            A = tf.nn.softmax(tf.matmul(Q_, tf.transpose(K_, perm=[0, 2, 1]))/math.sqrt(self.dim_V), axis=self.softmax_dim)
            
        qq= Q_ + tf.matmul(A,V_)
        O = tf.concat(tf.split(qq, math.ceil(np.shape(qq)[0]/np.shape(Q)[0]),0), axis=2)
        # S + GMH(S,H,A)
#         if getattr(self, 'ln0', None) != None:
#             O = self.ln0(O)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O) # z
        O = O + tf.nn.relu(self.fc_o(O))# z + rFF(z)
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)#GMPpool_k
        if return_attn:
            return O, A
        else:
            return O
        
    def get_fc_kv(self, dim_K, dim_V, conv):
        if conv == 'GCN':
            fc_k = GraphConv(dim_V, name='gfk1')
            fc_v = GraphConv(dim_V, name='gfk2')
            
        else:
            fc_k = Dense(dim_V, activation='linear', name='gfk3')
            fc_v = Dense(dim_V, activation='linear', name='gfk4')
        
        return fc_k, fc_v


class newMinCutPool(Layer):
    r"""
    A modified version of minCUT pooling layer as presented by
    [Bianchi et al. (2019)](https://arxiv.org/abs/1907.00481).
    """

    def __init__(self,
                 k,
                 temperature=1.0,
                 mlp_hidden=None,
                 mlp_activation='relu',
                 return_mask=False,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super().__init__(**kwargs)
        self.k = k
        self.temperature = temperature
        self.mlp_hidden = mlp_hidden if mlp_hidden else []
        self.mlp_activation = mlp_activation
        self.return_mask = return_mask
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint
        )
        mlp_layers = [
            Dense(self.k, 'linear', **layer_kwargs),
            Div2(self.temperature),
            Activation('softmax')
        ]
        self.mlp = Sequential(mlp_layers)

        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
            if K.ndim(I) == 2:
                I = I[:, 0]
        else:
            X, A = inputs
            I = None

        # Check if the layer is operating in batch mode (X and A have rank 3)
        batch_mode = K.ndim(X) == 3

        # Compute cluster assignment matrix
        S = self.mlp(X)

        # MinCut regularization
        A_pooled = ops.matmul_AT_B_A(S, A)
        num = tf.linalg.trace(A_pooled)
        D = ops.degree_matrix(A)
        den = tf.linalg.trace(ops.matmul_AT_B_A(S, D)) + K.epsilon()
        cut_loss = -(num / den)
        if batch_mode:
            cut_loss = K.mean(cut_loss)

        # Orthogonality regularization
        SS = ops.matmul_AT_B(S, S)
        I_S = tf.eye(self.k, dtype=SS.dtype)
        ortho_loss = tf.norm(
            SS / tf.norm(SS, axis=(-1, -2), keepdims=True) - I_S / tf.norm(I_S),
            axis=(-1, -2)
        )
        if batch_mode:
            ortho_loss = K.mean(ortho_loss)

        # Loss scaled ( tan(pi/6(Lmc + 1)) )
        Lmc = cut_loss + ortho_loss  # Lmc = Lc + Lo
        Lu = tf.math.tan(math.pi / 6 * (Lmc + 1))
        self.add_loss(Lu)

        # Pooling
        X_pooled = ops.matmul_AT_B(S, X)
        A_pooled = tf.linalg.set_diag(
            A_pooled, tf.zeros(K.shape(A_pooled)[:-1], dtype=A_pooled.dtype)
        )  # Remove diagonal
        A_pooled = ops.normalize_A(A_pooled)

        output = [X_pooled, A_pooled]

        if I is not None:
            I_mean = tf.math.segment_mean(I, I)
            I_pooled = ops.repeat(I_mean, tf.ones_like(I_mean) * self.k)
            output.append(I_pooled)

        if self.return_mask:
            output.append(S)

        return output


class Pruning(Layer):
    def __init__(self, X_hat_units, P, std=2.0, ind_not_active=0):
        super(Pruning, self).__init__()
        self.X_hat_units = X_hat_units
        self.P = P
        self.std = std
        self.ind_not_active = ind_not_active

    def call(self, inputs):
        X_hat, A_hat, X_in, A_in = inputs

        anomaly_score = self.P['anomaly_bias'] * tf.math.square(
            tf.norm(A_in - A_hat, axis=1, ord=2)) + \
                        (1 - self.P['anomaly_bias']) * tf.math.square(
            tf.norm((X_in - X_hat), axis=1, ord=2))

        mean = tf.math.reduce_mean(anomaly_score)
        std = tf.math.reduce_std(anomaly_score)

        I = tf.round(tf.nn.sigmoid(-1 * anomaly_score + mean + self.std * std + 1e-6))
        I = I - self.ind_not_active * I + self.ind_not_active
        X_pruned = tf.transpose(tf.transpose(X_in) * I)
        A_pruned = tf.transpose(tf.transpose(A_in)*I) * I

        loss_s = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(A_in, A_hat))
        loss_a = tf.reduce_mean(tf.math.square(X_in - X_hat))

        anomaly_loss = loss_s + loss_a
        self.add_loss(anomaly_loss)

        return X_pruned, A_pruned



class GraphGIN(GraphConv):

    def __init__(self,
                 channels,
                 epsilon=0,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(channels,
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        self.kernel_1 = self.add_weight(shape=(input_dim, self.channels),
                                        initializer=self.kernel_initializer,
                                        name='kernel_1',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.kernel_2 = self.add_weight(shape=(input_dim, self.channels),
                                        initializer=self.kernel_initializer,
                                        name='kernel_2',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features = inputs[0] #X
        fltr = inputs[1] #A

        # Convolution
        output = K.dot(features, self.kernel_1) #(X,W)
        output = ops.filter_dot(fltr, output) #(A,(X,W))

        # Skip connection
        skip = K.dot(features, self.kernel_2)
        output += skip * (1 + self.epsilon)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    @staticmethod
    def preprocess(A):
        return normalized_adjacency(A)
#%%

