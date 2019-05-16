# This file is used to preprocess uniprot and STRING file to get input for Graph2GO model

import pandas as pd
import numpy as np
import json
from networkx.readwrite import json_graph
import networkx as nx
import re
from collections import defaultdict
from scipy import sparse

def preprocessing(uniprot_file, string_file):
    print("Start preprocessing...")

    #------------------------------------------------
    # process uniprot
    def process_uniprot(uniprot_file):
        uniprot = pd.read_table(uniprot_file)

        # filter by STRING
        uniprot = uniprot[~uniprot['Cross-reference (STRING)'].isna()]
        uniprot.index = range(uniprot.shape[0])
        uniprot['Cross-reference (STRING)'] = uniprot['Cross-reference (STRING)'].apply(lambda x:x[:-1])

        # get GO labels
        uniprot['Gene ontology (biological process)'][uniprot['Gene ontology (biological process)'].isna()] = ''
        uniprot['Gene ontology (cellular component)'][uniprot['Gene ontology (cellular component)'].isna()] = ''
        uniprot['Gene ontology (molecular function)'][uniprot['Gene ontology (molecular function)'].isna()] = ''
        
        def get_GO(x):
            pattern = re.compile(r"GO:\d+")
            return pattern.findall(x)

        uniprot['cc'] = uniprot['Gene ontology (cellular component)'].apply(get_GO)
        uniprot['bp'] = uniprot['Gene ontology (biological process)'].apply(get_GO)
        uniprot['mf'] = uniprot['Gene ontology (molecular function)'].apply(get_GO)

        # filter GO by the number of occurence
        print("Filter GO by number of occurence...")
        cc = uniprot['cc'].values
        mf = uniprot['mf'].values
        bp = uniprot['bp'].values

        cc_count = defaultdict(int)
        mf_count = defaultdict(int)
        bp_count = defaultdict(int)

        for x in cc:
            for go in x:
                cc_count[go] += 1
        for x in mf:
            for go in x:
                mf_count[go] += 1
        for x in bp:
            for go in x:
                bp_count[go] += 1

        cc_list_10 = set()
        mf_list_10 = set()
        bp_list_30 = set()

        # If you use a new dataset, these threshold numbers may be changed
        for key,value in cc_count.items():
            if value >= 10:
                cc_list_10.add(key)
        for key,value in mf_count.items():
            if value >= 10:
                mf_list_10.add(key)
        for key,value in bp_count.items():
            if value >= 30:
                bp_list_30.add(key)

        # filter GO
        filter_cc = []
        filter_mf = []
        filter_bp = []

        for i,row in uniprot.iterrows():
            filter_cc.append(list(set(row['cc'])&cc_list_10))
            filter_mf.append(list(set(row['mf'])&mf_list_10))
            filter_bp.append(list(set(row['bp'])&bp_list_30))
        uniprot['filter_cc'] = filter_cc
        uniprot['filter_mf'] = filter_mf
        uniprot['filter_bp'] = filter_bp

        # generate label vectors
        cc_list_10 = list(cc_list_10)
        mf_list_10 = list(mf_list_10)
        bp_list_30 = list(bp_list_30)

        # write out filtered ontology lists
        def write_go_list(ontology,ll):
            with open("../../data/"+ontology+"_list.txt",'w') as f:
                for x in ll:
                    f.write(x + '\n')
        write_go_list('cc',cc_list_10)
        write_go_list('mf',mf_list_10)
        write_go_list('bp',bp_list_30)

        cc_mapping = dict(zip(cc_list_10,range(len(cc_list_10))))
        mf_mapping = dict(zip(mf_list_10,range(len(mf_list_10))))
        bp_mapping = dict(zip(bp_list_30,range(len(bp_list_30))))

        label_cc = np.zeros((len(uniprot),len(cc_list_10)))
        label_mf = np.zeros((len(uniprot),len(mf_list_10)))
        label_bp = np.zeros((len(uniprot),len(bp_list_30)))

        for i,row in uniprot.iterrows():
            for go in row['filter_cc']:
                label_cc[i][ cc_mapping[go] ] = 1
            for go in row['filter_mf']:
                label_mf[i][ mf_mapping[go] ] = 1
            for go in row['filter_bp']:
                label_bp[i][ bp_mapping[go] ] = 1

        uniprot['cc_label'] = list(label_cc)
        uniprot['mf_label'] = list(label_mf)
        uniprot['bp_label'] = list(label_bp)

        # filter by sequence length in order to compare with DeepGO
        uniprot['Length'] = uniprot['Sequence'].apply(len)
        uniprot = uniprot[ uniprot['Length'] <= 1000 ]

        # filter by ambiguous amino acid
        def find_amino_acid(x):
            return ('B' in x) | ('O' in x) | ('J' in x) | ('U' in x) | ('X' in x) | ('Z' in x)

        ambiguous_index = uniprot.loc[uniprot['Sequence'].apply(find_amino_acid)].index

        uniprot.drop(ambiguous_index, axis=0, inplace=True)
        uniprot.index = range(len(uniprot))

        #################################################
        ### encode amino acid sequence using CT encoding
        #################################################
        print("Encoding amino acid sequence...")
        def CT(sequence):
            classMap = {'G':'1','A':'1','V':'1','L':'2','I':'2','F':'2','P':'2',
                    'Y':'3','M':'3','T':'3','S':'3','H':'4','N':'4','Q':'4','W':'4',
                    'R':'5','K':'5','D':'6','E':'6','C':'7'}

            seq = ''.join([classMap[x] for x in sequence])
            length = len(seq)
            coding = np.zeros(343,dtype=np.int)
            for i in range(length-2):
                index = int(seq[i]) + (int(seq[i+1])-1)*7 + (int(seq[i+2])-1)*49 - 1
                coding[index] = coding[index] + 1
            return coding

        CT_list = []
        for seq in uniprot['Sequence'].values:
            CT_list.append(CT(seq))
        uniprot['CT'] = CT_list

        # Delete and rename some columns
        uniprot.drop(['Gene ontology (biological process)','Gene ontology (cellular component)',
              'Gene ontology (molecular function)','cc','mf','bp'],axis=1,inplace=True)
        uniprot['cc'] = uniprot['cc_label']
        uniprot['mf'] = uniprot['mf_label']
        uniprot['bp'] = uniprot['bp_label']
        uniprot.drop(['cc_label','mf_label','bp_label'],axis=1,inplace=True)


        ################################################
        ### encode subcellular location
        ################################################
        print("Encoding subcellular location...")
        def process_sub_loc(x):
            if str(x) == 'nan':
                return []
            x = x[22:-1]
            # check if exists "Note="
            pos = x.find("Note=")
            if pos != -1:
                x = x[:(pos-2)]
            temp = [t.strip() for t in x.split(".")]
            temp = [t.split(";")[0] for t in temp]
            temp = [t.split("{")[0].strip() for t in temp]
            temp = [x for x in temp if '}' not in x and x != '']
            return temp

        uniprot['Sub_cell_loc'] = uniprot['Subcellular location [CC]'].apply(process_sub_loc)
        items = [item for sublist in uniprot['Sub_cell_loc'] for item in sublist]
        items = np.unique(items)
        sub_mapping = dict(zip(list(items),range(len(items))))
        sub_encoding = [[0]*len(items) for i in range(len(uniprot))]
        for i,row in uniprot.iterrows():
            for loc in row['Sub_cell_loc']:
                sub_encoding[i][ sub_mapping[loc] ] = 1
        uniprot['Sub_cell_loc_encoding'] = sub_encoding
        uniprot.drop(['Subcellular location [CC]'],axis=1,inplace=True)

        ################################################
        ### encode protein domain
        ################################################
        print("Encoding protein domain...")
        def process_domain(x):
            if str(x) == 'nan':
                return []
            temp = [t.strip() for t in x[:-1].split(";")]
            return temp

        uniprot['protein-domain'] = uniprot['Cross-reference (Pfam)'].apply(process_domain)
        items = [item for sublist in uniprot['protein-domain'] for item in sublist]
        unique_elements, counts_elements = np.unique(items, return_counts=True)
        items = unique_elements[np.where(counts_elements > 5)]
        pro_mapping = dict(zip(list(items),range(len(items))))
        pro_encoding = [[0]*len(items) for i in range(len(uniprot))]

        for i,row in uniprot.iterrows():
            for fam in row['protein-domain']:
                if fam in pro_mapping:
                    pro_encoding[i][ pro_mapping[fam] ] = 1

        uniprot['Pro_domain_encoding'] = pro_encoding


        # save to a pickle file
        uniprot.to_pickle("../../data/features.pkl")
        print("\nWrite features into ../../data/features.pkl\n")


    #------------------------------------------------
    # process STRING interaction file
    def process_STRING(string_file):
        string = pd.read_table(string_file, delimiter=" ")
        uniprot = pd.read_pickle("../../data/features.pkl")

        # filter by uniprot
        string = string[string['protein1'].isin(uniprot['Cross-reference (STRING)'].values)]
        string = string[string['protein2'].isin(uniprot['Cross-reference (STRING)'].values)]

        # filter by confidence
        filtered = string[ string['combined_score']>400 ]

        # construct graph
        id_mapping = dict(zip(list(uniprot['Cross-reference (STRING)'].values),range(len(uniprot))))
        filtered['protein1_id'] = filtered['protein1'].apply(lambda x:id_mapping[x])
        filtered['protein2_id'] = filtered['protein2'].apply(lambda x:id_mapping[x])

        adj = np.zeros((len(id_mapping),len(id_mapping)))
        for i,row in filtered.iterrows():
            adj[row['protein1_id'],row['protein2_id']] = 1
        adj = sparse.csr_matrix(adj)
        sparse.save_npz("../../data/ppi_400.npz", adj)
        print("\nWrite interactions into ../../data/graph.npz\n")



    print("Process uniprot...")
    process_uniprot(uniprot_file)
    print("Process STRING...")
    process_STRING(string_file)


if __name__ == "__main__":
    preprocessing("../../data/uniprot.tab","../../data/string.txt")
