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

from evaluation import evaluate_performance, get_label_frequency

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
        #np.save(args.data_path + graph + "_" + str(attr) + "_embeddings.npy", embeddings)
    
    embeddings = np.hstack(embeddings_list)
    #np.save(os.path.join(args.data_path, args.species, "embeddings.npy"), embeddings)
    
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
    
    print("###################################")
    print('----------------------------------')
    print('CC')
    perf_cc = defaultdict(dict)
    index_11_30_cc, index_31_100_cc, index_101_300_cc, index_301_cc = get_label_frequency(cc)
    
    if args.supervised == "svm":
        y_score_cc = train_svm(X_train,Y_train_cc,X_test,Y_test_cc,"cc")
        perf_cc['11-30'] = evaluate_performance(Y_test_cc[:,index_11_30_cc], y_score_cc[:,index_11_30_cc])
        perf_cc['31-100'] = evaluate_performance(Y_test_cc[:,index_31_100_cc], y_score_cc[:,index_31_100_cc])
        perf_cc['101-300'] = evaluate_performance(Y_test_cc[:,index_101_300_cc], y_score_cc[:,index_101_300_cc])
        perf_cc['301-'] = evaluate_performance(Y_test_cc[:,index_301_cc], y_score_cc[:,index_301_cc])
        perf_cc['all'] = evaluate_performance(Y_test_cc, y_score_cc)
    elif args.supervised == "nn":
        y_score_11_30_cc = train_nn(X_train,Y_train_cc[:,index_11_30_cc],X_test,Y_test_cc[:,index_11_30_cc],"cc")
        perf_cc['11-30'] = evaluate_performance(Y_test_cc[:,index_11_30_cc], y_score_11_30_cc)
        y_score_31_100_cc = train_nn(X_train,Y_train_cc[:,index_31_100_cc],X_test,Y_test_cc[:,index_31_100_cc],"cc")
        perf_cc['31-100'] = evaluate_performance(Y_test_cc[:,index_31_100_cc], y_score_31_100_cc)
        y_score_101_300_cc = train_nn(X_train,Y_train_cc[:,index_101_300_cc],X_test,Y_test_cc[:,index_101_300_cc],"cc")
        perf_cc['101-300'] = evaluate_performance(Y_test_cc[:,index_101_300_cc], y_score_101_300_cc)
    print perf_cc
    
    filename = os.path.join(args.data_path, args.species, "results/" + "results_cc_graph2go_" + args.supervised + "_" + ";".join(args.graphs) + "_" + str(args.ppi_attributes) + "_" + str(args.simi_attributes) + "_" + str(args.thr_combined) + "_" + str(args.thr_evalue) + ".json")
    if args.test_flag:
        filename = os.path.join(args.data_path, args.species, "results/test_results", args.test_path,  "results_cc_graph2go_" + args.supervised + "_" + ";".join(args.graphs) + "_" + str(args.ppi_attributes) + "_" + str(args.simi_attributes) + "_" + str(args.thr_combined) + "_" + str(args.thr_evalue) + "_" + str(args.lr) + ".json")
    if args.save_results:
        with open(filename, "w") as f:
            json.dump(perf_cc, f)
    
    
    print('----------------------------------')
    print('MF')
    perf_mf = defaultdict(dict)
    index_11_30_mf, index_31_100_mf, index_101_300_mf, index_301_mf = get_label_frequency(mf)
    
    if args.supervised == "svm":
        y_score_mf = train_svm(X_train,Y_train_mf,X_test,Y_test_mf,"mf")
        perf_mf['11-30'] = evaluate_performance(Y_test_mf[:,index_11_30_mf], y_score_mf[:,index_11_30_mf])
        perf_mf['31-100'] = evaluate_performance(Y_test_mf[:,index_31_100_mf], y_score_mf[:,index_31_100_mf])
        perf_mf['101-300'] = evaluate_performance(Y_test_mf[:,index_101_300_mf], y_score_mf[:,index_101_300_mf])
        perf_mf['301-'] = evaluate_performance(Y_test_mf[:,index_301_mf], y_score_mf[:,index_301_mf])
        perf_mf['all'] = evaluate_performance(Y_test_mf, y_score_mf)
    elif args.supervised == "nn":
        y_score_11_30_mf = train_nn(X_train,Y_train_mf[:,index_11_30_mf],X_test,Y_test_mf[:,index_11_30_mf],"mf")
        perf_mf['11-30'] = evaluate_performance(Y_test_mf[:,index_11_30_mf], y_score_11_30_mf)
        y_score_31_100_mf = train_nn(X_train,Y_train_mf[:,index_31_100_mf],X_test,Y_test_mf[:,index_31_100_mf],"mf")
        perf_mf['31-100'] = evaluate_performance(Y_test_mf[:,index_31_100_mf], y_score_31_100_mf)
        y_score_101_300_mf = train_nn(X_train,Y_train_mf[:,index_101_300_mf],X_test,Y_test_mf[:,index_101_300_mf],"mf")
        perf_mf['101-300'] = evaluate_performance(Y_test_mf[:,index_101_300_mf], y_score_101_300_mf)
    print perf_mf
    
    filename = os.path.join(args.data_path, args.species, "results/" + "results_mf_graph2go_" + args.supervised + "_" + ";".join(args.graphs) + "_" + str(args.ppi_attributes) + "_" + str(args.simi_attributes) + "_" + str(args.thr_combined) + "_" + str(args.thr_evalue) + ".json")
    if args.test_flag:
        filename = os.path.join(args.data_path, args.species, "results/test_results", args.test_path,  "results_mf_graph2go_" + args.supervised + "_" + ";".join(args.graphs) + "_" + str(args.ppi_attributes) + "_" + str(args.simi_attributes) + "_" + str(args.thr_combined) + "_" + str(args.thr_evalue) + "_" + str(args.lr) + ".json")
    if args.save_results:
        with open(filename,"w") as f:
            json.dump(perf_mf, f)
    
    
    print('----------------------------------')
    print('BP')
    perf_bp = defaultdict(dict)
    index_11_30_bp, index_31_100_bp, index_101_300_bp, index_301_bp = get_label_frequency(bp)
    if args.supervised == "svm":
        y_score_bp = train_svm(X_train,Y_train_bp,X_test,Y_test_bp,"bp")
        perf_bp['11-30'] = evaluate_performance(Y_test_bp[:,index_11_30_bp], y_score_bp[:,index_11_30_bp])
        perf_bp['31-100'] = evaluate_performance(Y_test_bp[:,index_31_100_bp], y_score_bp[:,index_31_100_bp])
        perf_bp['101-300'] = evaluate_performance(Y_test_bp[:,index_101_300_bp], y_score_bp[:,index_101_300_bp])
        perf_bp['301-'] = evaluate_performance(Y_test_bp[:,index_301_bp], y_score_bp[:,index_301_bp])
        perf_bp['all'] = evaluate_performance(Y_test_bp, y_score_bp)
    elif args.supervised == "nn":
        y_score_11_30_bp = train_nn(X_train,Y_train_bp[:,index_11_30_bp],X_test,Y_test_bp[:,index_11_30_bp],"bp")
        perf_bp['11-30'] = evaluate_performance(Y_test_bp[:,index_11_30_bp], y_score_11_30_bp)
        y_score_31_100_bp = train_nn(X_train,Y_train_bp[:,index_31_100_bp],X_test,Y_test_bp[:,index_31_100_bp],"bp")
        perf_bp['31-100'] = evaluate_performance(Y_test_bp[:,index_31_100_bp], y_score_31_100_bp)
        y_score_101_300_bp = train_nn(X_train,Y_train_bp[:,index_101_300_bp],X_test,Y_test_bp[:,index_101_300_bp],"bp")
        perf_bp['101-300'] = evaluate_performance(Y_test_bp[:,index_101_300_bp], y_score_101_300_bp)
    print perf_bp
    
    filename = os.path.join(args.data_path, args.species, "results/" + "results_bp_graph2go_" + args.supervised + "_" + ";".join(args.graphs) + "_" + str(args.ppi_attributes) + "_" + str(args.simi_attributes) + "_" + str(args.thr_combined) + "_" + str(args.thr_evalue) + ".json")
    if args.test_flag:
        filename = os.path.join(args.data_path, args.species, "results/test_results", args.test_path,  "results_bp_graph2go_" + args.supervised + "_" + ";".join(args.graphs) + "_" + str(args.ppi_attributes) + "_" + str(args.simi_attributes) + "_" + str(args.thr_combined) + "_" + str(args.thr_evalue) + "_" + str(args.lr) + ".json")
    if args.save_results:
        with open(filename,"w") as f:
            json.dump(perf_bp, f)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #global parameters
    parser.add_argument('--ppi_attributes', type=int, default=6, help="types of attributes used by ppi.")
    parser.add_argument('--simi_attributes', type=int, default=5, help="types of attributes used by simi.")
    parser.add_argument('--graphs', type=lambda s:[item for item in s.split(",")], default=['combined','similarity'], help="lists of graphs to use.")    
    parser.add_argument('--species', type=str, default="human", help="which species to use.")
    parser.add_argument('--ontology', type=str, default="cc", help="which ontology to predict.")
    parser.add_argument('--data_path', type=str, default="../../data/", help="path storing data.")
    parser.add_argument('--thr_combined', type=float, default=0.3, help="threshold for combiend ppi network.")
    parser.add_argument('--thr_single', type=float, default=0.1, help="threshold for each single ppi network.")
    parser.add_argument('--thr_evalue', type=float, default=1e-4, help="threshold for similarity network.")
    parser.add_argument('--supervised', type=str, default="nn", help="neural networks or svm")
    parser.add_argument('--only_gcn', type=int, default=0, help="0 for training all, 1 for only embeddings.")
    parser.add_argument('--test_flag', type=int, default=0, help="whether it is a paramether test or not")
    parser.add_argument('--save_results', type=int, default=1, help="whether to save the performance results")
    parser.add_argument('--test_path', type=str, default='./', help="path for storing test results")
    
    
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
