import numpy as np
import pickle as pkl
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
from sklearn.preprocessing import scale,normalize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import json


def load_data(network_type="ppi", seq = False, attribute = 6):
    
    print('loading data...')
    
    def reshape(features):
        return np.hstack(features).reshape((len(features),len(features[0])))
    
    # get feature representations
    uniprot = pd.read_pickle("../../data/features.pkl")   
    features_seq = scale(reshape(uniprot['CT'].values))
    features_loc = reshape(uniprot['Sub_cell_loc_encoding'].values)
    features_domain = reshape(uniprot['Pro_domain_encoding'].values)
    
    print('generating features...')

    if attribute == 0:
        features = sp.identity(uniprot.shape[0]) 
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
    elif attribute == 6:
        print("Use all the features")
        if seq == True:
            features = np.concatenate((features_seq, features_loc,features_domain),axis=1)
            print("Including sequence feature")
        else:
            features = np.concatenate((features_loc,features_domain),axis=1)
    
    if attribute != 0:
        features = sp.csr_matrix(features)

    print('loading graph...')
    print(network_type)
    if network_type == "ppi":
        adj = sp.load_npz("../../data/ppi_400.npz")
    elif network_type == "similarity":
        adj = sp.load_npz("../../data/similarity_graph.npz")
    
    print('loading labels...')
    # load labels (GO)
    cc = uniprot['cc'].values
    cc = np.hstack(cc).reshape((len(cc),len(cc[0])))

    bp = uniprot['bp'].values
    bp = np.hstack(bp).reshape((len(bp),len(bp[0])))

    mf = uniprot['mf'].values
    mf = np.hstack(mf).reshape((len(mf),len(mf[0])))
    

    return adj, features, cc, bp, mf
