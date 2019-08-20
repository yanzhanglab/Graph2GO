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
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_gcn(features, adj_train, args, graph_type):
    model_str = args.model

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj_train
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj = adj_train

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
        model = GCNModelAE(placeholders, num_features, features_nonzero, args.hidden1, args.hidden2)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero, args.hidden1, args.hidden2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                          validate_indices=False), [-1]),
                          pos_weight=1,
                          norm=1,
                          lr=args.lr)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                           validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=1,
                           norm=1,
                           lr=args.lr)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)


    # Train model
    # use different epochs for ppi and similarity network
    if graph_type == "sequence_similarity":
        epochs = args.epochs_simi
    else:
        epochs = args.epochs_ppi

    for epoch in range(epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: args.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        
        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch+1), "train_loss=", "{:.5f}".format(outs[1]))


    print("Optimization Finished!")
    
    
    #return embedding for each protein
    emb = sess.run(model.z_mean,feed_dict=feed_dict)
    
    return emb

