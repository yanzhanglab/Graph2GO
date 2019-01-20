from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score
from optimizer import OptimizerAE, OptimizerVAE
from gcnModel import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_gcn(features, adj_train,  lr = 0.001, hidden1=800, hidden2=400, dropout=0, feature=1, epochs = 80):
    # Set up parameters
    flags = tf.app.flags    
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', lr, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', epochs, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', hidden1, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', hidden2, 'Number of units in hidden layer 2.')
    flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', dropout, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
    flags.DEFINE_integer('features', feature, 'Whether to use features (1) or not (0).')

    model_str = FLAGS.model

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj_train
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj = adj_train

    # featureless
    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float64),
        'adj': tf.sparse_placeholder(tf.float64),
        'adj_orig': tf.sparse_placeholder(tf.float64),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                          validate_indices=False), [-1]),
                          pos_weight=1,
                          norm=1)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                           validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=1,
                           norm=1)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

        print("Epoch:", '%04d' % (epoch+1), "train_loss=", "{:.5f}".format(outs[1]))


    print("Optimization Finished!")
    
    
    #return embedding for each protein
    emb = sess.run(model.z_mean,feed_dict=feed_dict)
    
    def del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            FLAGS.__delattr__(keys)
        
    del_all_flags(FLAGS)
    
    return emb

