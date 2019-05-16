from input_data import load_data
import numpy as np
import sys,getopt

from trainGcn import train_gcn
from trainNN import train_nn


def train(attribute):
    # load data for ppi
    adj_ppi, features, cc, bp, mf = load_data("ppi", True, attribute)
    embeddings_ppi = train_gcn(features,adj_ppi)
    
    np.save("../../data/ppi_embeddings.npy", embeddings_ppi)
    
    # load data for similarity network
    adj_simi, features, cc, bp, mf = load_data("similarity", False, attribute)
    embeddings_simi = train_gcn(features,adj_simi)
    
    np.save("../../data/simi_embeddings.npy", embeddings_simi)
    
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
    
    #X_train_ppi = embeddings_ppi[train_idx]
    #X_test_ppi = embeddings_ppi[test_idx]
    #X_train_simi = embeddings_simi[train_idx]
    #X_test_simi = embeddings_simi[test_idx]
    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]

    '''
    print("###################################")
    print("PPI")
    #train_nn(X_train,Y_train_cc,X_test,Y_test_cc)
    print('----------------------------------')
    print('CC')
    train_nn(X_train_ppi,Y_train_cc,X_test_ppi,Y_test_cc,"cc")    

    #train_nn(X_train,Y_train_mf,X_test,Y_test_mf)
    print('----------------------------------')
    print('MF')
    train_nn(X_train_ppi,Y_train_mf,X_test_ppi,Y_test_mf,"mf")   

    #train_nn(X_train,Y_train_bp,X_test,Y_test_bp)
    print('----------------------------------')
    print('BP')
    train_nn(X_train_ppi,Y_train_bp,X_test_ppi,Y_test_bp,"bp")
    
    
    print("###################################")
    print("Similarity")
    #train_nn(X_train,Y_train_cc,X_test,Y_test_cc)
    print('----------------------------------')
    print('CC')
    train_nn(X_train_simi,Y_train_cc,X_test_simi,Y_test_cc,"cc")   

    #train_nn(X_train,Y_train_mf,X_test,Y_test_mf)
    print('----------------------------------')
    print('MF')
    train_nn(X_train_simi,Y_train_mf,X_test_simi,Y_test_mf,"mf")   

    #train_nn(X_train,Y_train_bp,X_test,Y_test_bp)
    print('----------------------------------')
    print('BP')
    train_nn(X_train_simi,Y_train_bp,X_test_simi,Y_test_bp,"bp")
    '''
    
    print("###################################")
    print("Combined network")
    #train_nn(X_train,Y_train_cc,X_test,Y_test_cc)
    print('----------------------------------')
    print('CC')
    #train_nn(X_train,Y_train_cc,X_test,Y_test_cc,"cc")
    train_nn(embeddings,cc,embeddings,cc,"cc")

    #train_nn(X_train,Y_train_mf,X_test,Y_test_mf)
    print('----------------------------------')
    print('MF')
    #train_nn(X_train,Y_train_mf,X_test,Y_test_mf,"mf")
    train_nn(embeddings,mf,embeddings,mf,"mf")

    #train_nn(X_train,Y_train_bp,X_test,Y_test_bp)
    print('----------------------------------')
    print('BP')
    #train_nn(X_train,Y_train_bp,X_test,Y_test_bp,"bp")
    train_nn(embeddings,bp,embeddings,bp,"bp")
    

if __name__ == "__main__":

    opts, args = getopt.getopt(sys.argv[1:],"l:d:e:s:h:m",["learning_rate=","epochs=","attribute=","hidden1=","hidden2="])
    lr = 0.001
    epochs = 50
    # number of nodes in the hidden layer
    hidden1 = 512
    hidden2 = 256
    attribute = 6

    for opt, arg in opts:
        if opt == '--learning_rate':
            lr = float(arg)
        if opt == "--epochs":
            epochs = int(arg)
        if opt == "--attribute":
            attribute = int(arg)
        if opt == "--hidden1":
            hidden1 = int(hidden1)
        if opt == "--hidden2":
            hidden2 = int(hidden2)
    
    #train(lr, dropout, epochs, seq, hidden1, hidden2)
    train(attribute)
