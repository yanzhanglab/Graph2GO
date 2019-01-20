from input_data import load_data
import numpy as np
import sys,getopt

from trainGcn import train_gcn
from trainNN import train_nn


def train(lr, dropout, epochs):
    # load data
    adj, features, cc, bp, mf = load_data()

    # train GCN model to get embeddings for each protein
    embeddings = train_gcn(features,adj, lr=lr, dropout=dropout, epochs=epochs)
    
    np.random.seed(5959)

    # split data into train and test
    num_test = int(np.floor(cc.shape[0] / 5.))
    num_train = cc.shape[0] - num_test
    all_idx = list(range(cc.shape[0]))
    np.random.shuffle(all_idx)
    train_idx = all_idx[:num_train]
    X_train = embeddings[train_idx]
    Y_train_cc = cc[train_idx]
    Y_train_bp = bp[train_idx]
    Y_train_mf = mf[train_idx]
    test_idx = all_idx[num_train:(num_train + num_test)]
    X_test = embeddings[test_idx]
    Y_test_cc = cc[test_idx]
    Y_test_bp = bp[test_idx]
    Y_test_mf = mf[test_idx]

    #the final neural network classifier
    #train_nn(X_train,Y_train_cc,X_test,Y_test_cc)
    train_nn(X_train,Y_train_cc,X_test,Y_test_cc,"cc")
    print('----------------------------------')

    #train_nn(X_train,Y_train_mf,X_test,Y_test_mf)
    train_nn(X_train,Y_train_mf,X_test,Y_test_mf,"mf")
    print('----------------------------------')

    #train_nn(X_train,Y_train_bp,X_test,Y_test_bp)
    train_nn(X_train,Y_train_bp,X_test,Y_test_bp,"bp")
    print('----------------------------------')
    

if __name__ == "__main__":

    opts, args = getopt.getopt(sys.argv[1:],"l:d:e:",["lr=","dropout=","epochs="])
    lr = 0.001
    dropout = 0
    epochs = 80
    for opt, arg in opts:
        if opt == '--lr':
            lr = float(arg)
        if opt == '--dropout':
            dropout = float(arg)
        if opt == "--epochs":
            epochs = int(arg)
    print("learning rate:",lr)
    print("epochs:",epochs)
    
    train(lr, dropout, epochs)
