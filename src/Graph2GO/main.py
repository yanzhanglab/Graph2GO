from input_data import load_data,load_labels
import numpy as np
import pandas as pd
import sys,getopt
import argparse
import json
import os
from collections import defaultdict

from trainGcn import train_gcn
from trainNN import train_nn
from trainSVM import train_svm

from evaluation import get_results

def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))


def train(args):
    # load feature dataframe
    print("loading features...") 
    uniprot = pd.read_pickle(os.path.join(args.data_path, args.species, "features.pkl"))
    
    embeddings_list = []
    for graph in args.graphs:
        print("#############################")
        print("Training",graph)
        adj, features = load_data(graph, uniprot, args)
        embeddings = train_gcn(features, adj, args, graph)
        embeddings_list.append(embeddings)
        
        if graph == "combined":
            attr = args.ppi_attributes
        elif graph == "similarity":
            attr = args.simi_attributes
    
    embeddings = np.hstack(embeddings_list)
    
    if args.only_gcn == 1:
        return
    
    #load labels
    cc,mf,bp = load_labels(uniprot)
    
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

    print("Start running supervised model...")
    rand_str = np.random.randint(1000000)
    save_path = os.path.join(args.data_path, args.species, "results_new/results_graph2go_" + args.supervised + "_" + ";".join(args.graphs) + "_" + str(args.ppi_attributes) + "_" + str(args.simi_attributes) + "_" + str(args.thr_combined) + "_" + str(args.thr_evalue))
    
    print("###################################")
    print('----------------------------------')
    print('CC')
    
    if args.supervised == "svm":
        y_score_cc = train_svm(X_train,Y_train_cc,X_test,Y_test_cc)
    elif args.supervised == "nn":
        y_score_cc = train_nn(X_train,Y_train_cc,X_test,Y_test_cc)
    
    perf_cc = get_results(cc, Y_test_cc, y_score_cc)
    if args.save_results:
        with open(save_path + "_cc.json", "w") as f:
            json.dump(perf_cc, f)
    
    
    print('----------------------------------')
    print('MF')
    
    if args.supervised == "svm":
        y_score_mf = train_svm(X_train,Y_train_mf,X_test,Y_test_mf)
    elif args.supervised == "nn":
        y_score_mf = train_nn(X_train,Y_train_mf,X_test,Y_test_mf)
    
    perf_mf = get_results(mf, Y_test_mf, y_score_mf)
    if args.save_results:
        with open(save_path + "_mf.json","w") as f:
            json.dump(perf_mf, f)
    
    
    print('----------------------------------')
    print('BP')

    if args.supervised == "svm":
        y_score_bp = train_svm(X_train,Y_train_bp,X_test,Y_test_bp)
    elif args.supervised == "nn":
        y_score_bp = train_nn(X_train,Y_train_bp,X_test,Y_test_bp)
    
    perf_bp = get_results(bp, Y_test_bp, y_score_bp)
    if args.save_results:
        with open(save_path + "_bp.json","w") as f:
            json.dump(perf_bp, f)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #global parameters
    parser.add_argument('--ppi_attributes', type=int, default=6, help="types of attributes used by ppi.")
    parser.add_argument('--simi_attributes', type=int, default=5, help="types of attributes used by simi.")
    parser.add_argument('--graphs', type=lambda s:[item for item in s.split(",")], default=['combined','similarity'], help="lists of graphs to use.")    
    parser.add_argument('--species', type=str, default="human", help="which species to use.")
    parser.add_argument('--data_path', type=str, default="../../data/", help="path storing data.")
    parser.add_argument('--thr_combined', type=float, default=0.3, help="threshold for combiend ppi network.")
    parser.add_argument('--thr_evalue', type=float, default=1e-4, help="threshold for similarity network.")
    parser.add_argument('--supervised', type=str, default="nn", help="neural networks or svm")
    parser.add_argument('--only_gcn', type=int, default=0, help="0 for training all, 1 for only embeddings.")
    parser.add_argument('--save_results', type=int, default=1, help="whether to save the performance results")
    
    #parameters for traing GCN
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--epochs_ppi', type=int, default=80, help="Number of epochs to train ppi.")
    parser.add_argument('--epochs_simi', type=int, default=60, help="Number of epochs to train similarity network.")
    parser.add_argument('--hidden1', type=int, default=800, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=400, help="Number of units in hidden layer 2.")
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight for L2 loss on embedding matrix.")
    parser.add_argument('--dropout', type=float, default=0, help="Dropout rate (1 - keep probability).")
    parser.add_argument('--model', type=str, default="gcn_vae", help="Model string.")
    
    args = parser.parse_args()
    print(args)

    train(args)
