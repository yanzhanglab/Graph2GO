import numpy as np
import pickle as pkl
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
from sklearn.preprocessing import scale,normalize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import json
import os
from tqdm import tqdm


def load_ppi_network(filename, gene_num, thr):
    with open(filename) as f:
        data = f.readlines()
    adj = np.zeros((gene_num,gene_num))
    for x in tqdm(data):
        temp = x.strip().split("\t")
        # check whether score larger than the threshold
        if float(temp[2]) >= thr:
            adj[int(temp[0]),int(temp[1])] = 1
    if (adj.T == adj).all():
        pass
    else:
        adj = adj + adj.T
    
    return adj

def load_simi_network(filename, gene_num, thr):
    with open(filename) as f:
        data = f.readlines()
    adj = np.zeros((gene_num,gene_num))
    for x in tqdm(data):
        temp = x.strip().split("\t")
        # check whether evalue smaller than the threshold
        if float(temp[2]) <= thr:
            adj[int(temp[0]),int(temp[1])] = 1
    if (adj.T == adj).all():
        pass
    else:
        adj = adj + adj.T
    
    return adj

def load_labels(uniprot):
    print('loading labels...')
    # load labels (GO)
    cc = uniprot['cc_label'].values
    cc = np.hstack(cc).reshape((len(cc),len(cc[0])))

    bp = uniprot['bp_label'].values
    bp = np.hstack(bp).reshape((len(bp),len(bp[0])))

    mf = uniprot['mf_label'].values
    mf = np.hstack(mf).reshape((len(mf),len(mf[0])))
    
    return cc,mf,bp


def load_data(graph_type, uniprot, args):
    
    print('loading data...')
    
    def reshape(features):
        return np.hstack(features).reshape((len(features),len(features[0])))
    
    # get feature representations 
    features_seq = scale(reshape(uniprot['CT'].values))
    features_loc = reshape(uniprot['Sub_cell_loc_encoding'].values)
    features_domain = reshape(uniprot['Pro_domain_encoding'].values)
    
    print('generating features...')

    if graph_type == "ppi":
        attribute = args.ppi_attributes
    elif graph_type == "sequence_similarity":
        attribute = args.simi_attributes
        
    if attribute == 0:
        features = np.identity(uniprot.shape[0]) 
        print("Without features")
    elif attribute == 1:
        features = features_seq
        print("Only use sequence feature")
    elif attribute == 2:
        features = features_loc
        print("Only use location feature")
    elif attribute == 3:
        features = features_domain
        print("Only use domain feature")
    elif attribute == 5:
        features = np.concatenate((features_loc,features_domain),axis=1)
        print("use location and domain features")
    elif attribute == 6:
        features = np.concatenate((features_seq, features_loc,features_domain),axis=1)
        print("Use all the features")   
    elif attribute == 7:
        features = np.concatenate((features_seq, features_loc,features_domain, np.identity(uniprot.shape[0])),axis=1)
        print("Use all features plus identity")
    
    features = sp.csr_matrix(features)

    print('loading graph...')
    if graph_type == "sequence_similarity":
        filename = os.path.join(args.data_path, args.species, "networks/sequence_similarity.txt")
        adj = load_simi_network(filename, uniprot.shape[0], args.thr_evalue)
    elif graph_type == "ppi":
        filename = os.path.join(args.data_path, args.species, "networks/ppi.txt")
        adj = load_ppi_network(filename, uniprot.shape[0], args.thr_ppi)
    
    adj = sp.csr_matrix(adj)
     

    return adj, features
