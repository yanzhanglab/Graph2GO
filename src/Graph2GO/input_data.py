import numpy as np
import pickle as pkl
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
from sklearn.preprocessing import scale,normalize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def load_data():
    
    print('loading data...')
    
    def reshape(features):
        return np.hstack(features).reshape((len(features),len(features[0])))
    
    # get feature representations
    uniprot = pd.read_pickle("../../data/features.pkl")   
    features_seq = scale(reshape(uniprot['CT'].values))
    features_loc = reshape(uniprot['Sub_cell_loc_encoding'].values)
    features_domain = reshape(uniprot['Pro_domain_encoding'].values)
    
    # concatenate different features
    features = np.concatenate((features_seq, features_loc,features_domain),axis=1)
    features = sp.csr_matrix(features)

    adj = sp.load_npz("../../data/graph.npz")
    
    
    # load labels (GO)
    cc = uniprot['cc'].values
    cc = np.hstack(cc).reshape((len(cc),len(cc[0])))

    bp = uniprot['bp'].values
    bp = np.hstack(bp).reshape((len(bp),len(bp[0])))

    mf = uniprot['mf'].values
    mf = np.hstack(mf).reshape((len(mf),len(mf[0])))
    

    return adj, features, cc, bp, mf
