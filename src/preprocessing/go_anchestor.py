from collections import deque
import pandas as pd
from xml.etree import ElementTree as ET

def get_gene_ontology(filename='../../data/go-basic.obo'):
    # Reading Gene Ontology from OBO Formatted file
    go = dict()
    obj = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    go[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
    if obj is not None:
        go[obj['id']] = obj
    for go_id in go.keys():
        if go[go_id]['is_obsolete']:
            del go[go_id]
    for go_id, val in go.iteritems():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in go:
                if 'children' not in go[p_id]:
                    go[p_id]['children'] = set()
                go[p_id]['children'].add(go_id)
    return go


def get_anchestors(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while(len(q) > 0):
        g_id = q.popleft()
        go_set.add(g_id)
        if g_id not in go:
            #print g_id
            continue
        for parent_id in go[g_id]['is_a']:
            if parent_id in go:
                q.append(parent_id)
    return go_set


def get_parents(go, go_id):
    go_set = set()
    for parent_id in go[go_id]['is_a']:
        if parent_id in go:
            go_set.add(parent_id)
    return go_set


def get_go_set(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while len(q) > 0:
        g_id = q.popleft()
        go_set.add(g_id)
        for ch_id in go[g_id]['children']:
            q.append(ch_id)
    return go_set    
