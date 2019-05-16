from input_data import load_data
import numpy as np
import sys,getopt

from trainGcn import train_gcn
from trainNN import train_nn


def train(attribute):
    # load data for ppi
    adj_ppi, features, cc, bp, mf = load_data("ppi", True, attribute)
    print("Training PPI network...")
    embeddings_ppi = train_gcn(features,adj_ppi)
    
    # save ppi eembeddings
    #np.save("../../data/ppi_embeddings.npy", embeddings_ppi)
    
    # load data for similarity network
    adj_simi, features, cc, bp, mf = load_data("similarity", False, attribute)
    print("Training sequence similarity network...")
    embeddings_simi = train_gcn(features,adj_simi)
    
    # save similarity embeddings
    #np.save("../../data/simi_embeddings.npy", embeddings_simi)
    
    embeddings = np.hstack((embeddings_ppi, embeddings_simi))
    
    np.random.seed(5959)

    # split data into train and test
    num_test = int(np.floor(cc.shape[0] / 5.))
    num_train = cc.shape[0] - num_test
    all_idx = list(range(cc.shape[0]))
    np.random.shuffle(all_idx)
    
    train_idx = all_idx[:num_train]
    test_idx = all_idx[num_train:(num_train + num_test)]
    
    Y_train_cc = cc[train_idx]
    Y_train_bp = bp[train_idx]
    Y_train_mf = mf[train_idx]
    
    Y_test_cc = cc[test_idx]
    Y_test_bp = bp[test_idx]
    Y_test_mf = mf[test_idx]

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]

    print("###################################")
    print("Combined network")

    print('----------------------------------')
    print('CC')
    train_nn(X_train,Y_train_cc,X_test,Y_test_cc,"cc")


    print('----------------------------------')
    print('MF')
    train_nn(X_train,Y_train_mf,X_test,Y_test_mf,"mf")

    print('----------------------------------')
    print('BP')
    train_nn(X_train,Y_train_bp,X_test,Y_test_bp,"bp")
    

if __name__ == "__main__":
    # 6 means using all attributes in the network
    attribute = 6
    train(attribute)
