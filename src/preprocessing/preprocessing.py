# This file is used to preprocess uniprot and STRING file to get input for Graph2GO model

import pandas as pd
import numpy as np
import json
from networkx.readwrite import json_graph
import networkx as nx
import re
from collections import defaultdict
from scipy import sparse
import argparse
from tqdm import tqdm
import os

from go_anchestor import get_gene_ontology,get_anchestors

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default="../../data/", help="path storing data.")
parser.add_argument('--species', type=str, default="human", help="which species to use.")
args = parser.parse_args()


##########################################
########## process uniprot ###############

print("Start processing uniprot...")

#### load file
print("Loading data...")
uniprot_file = os.path.join(args.data_path, args.species, "uniprot-" + args.species + ".tab")
uniprot = pd.read_table(uniprot_file)
print(uniprot.shape)


#### filtering
print("filtering...")
# filter by STRING ID occurence
uniprot = uniprot[~uniprot['Cross-reference (STRING)'].isna()]
uniprot.index = range(uniprot.shape[0])
uniprot['Cross-reference (STRING)'] = uniprot['Cross-reference (STRING)'].apply(lambda x:x[:-1])
        
# filter by sequence length in order to compare with DeepGO
uniprot['Length'] = uniprot['Sequence'].apply(len)
uniprot = uniprot[ uniprot['Length'] <= 1000 ]

# filter by ambiguous amino acid
def find_amino_acid(x):
    return ('B' in x) | ('O' in x) | ('J' in x) | ('U' in x) | ('X' in x) | ('Z' in x)

ambiguous_index = uniprot.loc[uniprot['Sequence'].apply(find_amino_acid)].index
uniprot.drop(ambiguous_index, axis=0, inplace=True)
uniprot.index = range(len(uniprot))
print("after filtering:", uniprot.shape)



#### obtain GO annotations
print("obtain GO annotations...")
uniprot['Gene ontology (biological process)'][uniprot['Gene ontology (biological process)'].isna()] = ''
uniprot['Gene ontology (cellular component)'][uniprot['Gene ontology (cellular component)'].isna()] = ''
uniprot['Gene ontology (molecular function)'][uniprot['Gene ontology (molecular function)'].isna()] = ''

def get_GO(x):
    pattern = re.compile(r"GO:\d+")
    return pattern.findall(x)

uniprot['cc'] = uniprot['Gene ontology (cellular component)'].apply(get_GO)
uniprot['bp'] = uniprot['Gene ontology (biological process)'].apply(get_GO)
uniprot['mf'] = uniprot['Gene ontology (molecular function)'].apply(get_GO)

num_cc_before = sum(len(x) for x in uniprot['cc'])
num_mf_before = sum(len(x) for x in uniprot['mf'])
num_bp_before = sum(len(x) for x in uniprot['bp'])
print "number of CCs, before enrich", num_cc_before
print "number of MFs, before enrich", num_mf_before
print "number of BPs, before enrich", num_bp_before



print("start enriching go annotations...")
# enrich go terms using ancestors
go = get_gene_ontology(os.path.join(args.data_path, "go-basic.obo"))
BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'

new_cc = []
new_mf = []
new_bp = []

for i, row in uniprot.iterrows():
    labels = row['cc']
    temp = set([])
    for x in labels:
        temp = temp | get_anchestors(go, x)
    temp.discard(CELLULAR_COMPONENT)
    new_cc.append(list(temp))

    labels = row['mf']
    temp = set([])
    for x in labels:
        temp = temp | get_anchestors(go, x)
    temp.discard(MOLECULAR_FUNCTION)
    new_mf.append(list(temp))

    labels = row['bp']
    temp = set([])
    for x in labels:
        temp = temp | get_anchestors(go, x)
    temp.discard(BIOLOGICAL_PROCESS)
    new_bp.append(list(temp))

uniprot['cc'] = new_cc
uniprot['mf'] = new_mf
uniprot['bp'] = new_bp

num_cc_after = sum(len(x) for x in uniprot['cc'])
num_mf_after = sum(len(x) for x in uniprot['mf'])
num_bp_after = sum(len(x) for x in uniprot['bp'])
print "number of CCs, after enrich", num_cc_after
print "number of MFs, after enrich", num_mf_after
print "number of BPs, after enrich", num_bp_after



#### filter GO terms by the number of occurence
print("filter GO terms by the number of occurence...")
# filter GO by the number of occurence
mf_items = [item for sublist in uniprot['mf'] for item in sublist]
mf_unique_elements, mf_counts_elements = np.unique(mf_items, return_counts=True)
bp_items = [item for sublist in uniprot['bp'] for item in sublist]
bp_unique_elements, bp_counts_elements = np.unique(bp_items, return_counts=True)
cc_items = [item for sublist in uniprot['cc'] for item in sublist]
cc_unique_elements, cc_counts_elements = np.unique(cc_items, return_counts=True)

mf_list = mf_unique_elements[np.where(mf_counts_elements >= 10)]
cc_list = cc_unique_elements[np.where(cc_counts_elements >= 10)]
bp_list = bp_unique_elements[np.where(bp_counts_elements >= 10)]

temp_mf = uniprot['mf'].apply(lambda x: list(set(x) & set(mf_list)))
uniprot['filter_mf'] = temp_mf
temp_cc = uniprot['cc'].apply(lambda x: list(set(x) & set(cc_list)))
uniprot['filter_cc'] = temp_cc
temp_bp = uniprot['bp'].apply(lambda x: list(set(x) & set(bp_list)))
uniprot['filter_bp'] = temp_bp

# write out filtered ontology lists
def write_go_list(ontology,ll):
    filename = os.path.join(args.data_path, args.species, ontology+"_list.txt")
    with open(filename,'w') as f:
        for x in ll:
            f.write(x + '\n')
print("writing go term list...")
write_go_list('cc',cc_list)
write_go_list('mf',mf_list)
write_go_list('bp',bp_list)



#### encode GO terms
print("encoding GO terms...")
mf_dict = dict(zip(list(mf_list),range(len(mf_list))))
cc_dict = dict(zip(list(cc_list),range(len(cc_list))))
bp_dict = dict(zip(list(bp_list),range(len(bp_list))))
mf_encoding = [[0]*len(mf_dict) for i in range(len(uniprot))]
cc_encoding = [[0]*len(cc_dict) for i in range(len(uniprot))]
bp_encoding = [[0]*len(bp_dict) for i in range(len(uniprot))]

for i,row in uniprot.iterrows():
    for x in row['filter_mf']:
        mf_encoding[i][ mf_dict[x] ] = 1
    for x in row['filter_cc']:
        cc_encoding[i][ cc_dict[x] ] = 1
    for x in row['filter_bp']:
        bp_encoding[i][ bp_dict[x] ] = 1

uniprot['cc_label'] = cc_encoding
uniprot['mf_label'] = mf_encoding
uniprot['bp_label'] = bp_encoding

uniprot.drop(columns=['mf','cc','bp','Gene ontology (biological process)',
                      'Gene ontology (cellular component)',
                      'Gene ontology (molecular function)'],inplace=True)



#### encode amino acid sequence using CT
print("encode amino acid sequence using CT...")
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


#### encode subcellular location
print("encode subcellular location...")

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


#### encode protein domains
print("encode protein domains...")

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


#### wirte files
print("write files...")
uniprot.to_pickle(os.path.join(args.data_path, args.species, "features.pkl"))
uniprot[['Entry','Gene names','Cross-reference (STRING)']].to_csv(os.path.join(args.data_path,args.species,"gene_list.csv"),
                                                                 index_label='ID')



#################################
######## process PPIs ###########

print("Start processing PPIs...")

string_file = os.path.join(args.data_path, args.species, "string-"+args.species+".txt")
string = pd.read_table(string_file, delimiter=" ")
gene_list = pd.read_csv(os.path.join(args.data_path,args.species,"gene_list.csv"))

# filter by uniprot
string = string[string['protein1'].isin(gene_list['Cross-reference (STRING)'].values)]
string = string[string['protein2'].isin(gene_list['Cross-reference (STRING)'].values)]

# map names to indexs
id_mapping = dict(zip(list(gene_list['Cross-reference (STRING)'].values), 
                     list(gene_list['ID'].values)))
string['protein1_id'] = string['protein1'].apply(lambda x:id_mapping[x])
string['protein2_id'] = string['protein2'].apply(lambda x:id_mapping[x])

subnetwork = string[['protein1_id','protein2_id','combined_score']]
subnetwork['combined_score'] = subnetwork['combined_score']/1000.0
subnetwork.to_csv(os.path.join(args.data_path, args.species, "networks/ppi.txt"), index=False, header=False, sep="\t")



###################################
######## process similarity #######

print("Start processing similarity...")

def write_fasta(feats):
    filename = os.path.join(args.data_path, args.species, "blast/uniprot_seq.fas")
    with open(filename, "w") as f:
        for i,row in feats.iterrows():
            f.write(">" + row['Entry'] + "\n")
            f.write(row['Sequence'] + "\n")  

write_fasta(uniprot)