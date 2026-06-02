#region libraries

import os
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering
import hdbscan
from openai import OpenAI

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
import gc
from rapidfuzz import fuzz as fuzzy

#endregion

#region make distance matrix

def make_zigzag_list(items):
    listzig,i=[],0
    numitems=len(items)
    min, max=0, numitems-1
    while min<max:
        listzig+=[items[min],items[max]]
        i+=1
        min,max=i, numitems-1-i
    return listzig

def fuzzy_scores(row_names, indices):
    results=[]
    for i in indices:
        for j in range(i+1, len(row_names)):
            results.append((i, j, float(fuzzy.WRatio(row_names[i], row_names[j])))) #0.00015 s per pair
        # result=np.hstack((np.array([i]*len(fuzzylist)).reshape(-1,1), np.arange(i+1, len(row_names)).reshape(-1,1), np.array(fuzzylist, dtype=float).reshape(-1,1))).reshape(-1,3).astype(object)
        # results.append(result)
    return(np.vstack(results))

def bestfuzz(lista,listb):
    best = 0
    for a in lista:
        for b in listb:
            s = fuzzy.WRatio(a, b)
            if s > best:
                best = s
                if best == 100:
                    return 100
    return best

def multi_fuzzy_scores(row_lists, indices):
    results=[]
    row_sets = [set(row) for row in row_lists]
    for i in indices:
        for j in range(i+1, len(row_sets)):
            if row_sets[i] & row_sets[j]:
                results.append((i, j, float(100)))
            else:
                fuzzyscore=bestfuzz(row_sets[i], row_sets[j])
                results.append((i, j, float(fuzzyscore)))
    return(np.array(results, dtype=object))

def fuzzy_scores_save(prefix, row_names, indices, path):
    subbatches=split_into_batches(indices,10,'n_batches')
    for z, batch in enumerate(subbatches):
        results=[]
        for i in batch:
            fuzzylist=[fuzz.WRatio(row_names[i], row_names[j]) for j in range(i+1, len(row_names))] #0.00015 s per pair
            result=np.hstack((np.array([i]*len(fuzzylist)).reshape(-1,1), np.arange(i+1, len(row_names)).reshape(-1,1), np.array(fuzzylist, dtype=float).reshape(-1,1))).reshape(-1,3).astype(object)
            results.append(result)
        save_file(np.vstack(results), path, f'{prefix}_{z}')
        del results
        gc.collect()

def populate_matrix(fuzzy_scores, n_rows):
    rows, cols, data = zip(*fuzzy_scores)
    matrix=coo_matrix((data, (rows, cols)), shape=(n_rows, n_rows)).tocsr()
    matrix=matrix + matrix.T
    matrix.setdiag(100)
    matrix = matrix.toarray()
    matrix = 1.0 - matrix / 100.0
    return matrix

#endregion

#region hierarchical and density clustering


"""
agglomerative clustering:

density clustering: 
computes core distances, distance to k-nearest neighbour (k=min_samples), 
calculate mutual reachability distances for each 2 points max(core_distance(A), core_distance(B), distance(A, B)),
build minimum spanning tree (uses mutual reachability distances as edge weights)--> connect all points with least total edge weight
remove edges (epsilon plays a role here)
identify clusters
"""
def run_hierarchical(obj, num_clusters, distance, linktype):
    """
    obj is distance matrix if linktype is not ward, else its the locations of every point (embedding location)
    """
    if linktype!='ward':
        model= AgglomerativeClustering(n_clusters=num_clusters, distance_threshold=distance, linkage=linktype, metric='precomputed')
        model.fit(obj)
    else:
        model = AgglomerativeClustering(n_clusters=num_clusters, distance_threshold=distance, linkage=linktype)
        model.fit(obj)
    return model.labels_


def run_density(obj, epsilon):
    """
    obj is distance matrix
    """
    model = hdbscan.HDBSCAN(min_cluster_size=2,cluster_selection_epsilon=float(epsilon), gen_min_span_tree=False, metric='precomputed')
    model.fit(obj)
    clusters= model.labels_  #get cluster labels
    recursive_max= np.max(clusters) + 1  #get max cluster label
    for i in np.where(clusters == -1)[0]:
        clusters[i]=recursive_max
        recursive_max+=1
    return(clusters)

#endregion

#region openai embeddings

def parallel_get_embedding(text_col, n_workers, model="text-embedding-3-small"):
    """
    Takes a column of text and returns embeddings for each text in the column
    """
    def get_embedding(client, text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding

    def subprocess(indices, text_col, model):
        gpt_key=os.getenv('openai_key')  #get OpenAI API key from environment variable
        client = OpenAI()
        embeddings = np.array([get_embedding(client, text_col[i], model) for i in indices])
        chunk= np.hstack((indices.reshape(-1,1), embeddings))  #add indices to embeddings
        return chunk
    
    batches= np.array_split(np.arange(len(text_col)), min(n_workers, len(text_col)))  # split text column into batches
    results = Parallel(n_jobs=n_workers, backend='loky')(delayed(subprocess)(batch, text_col, model) for batch in batches)
    results=np.vstack(results)
    results=results[np.argsort(results[:, 0]), :]  #sort results by indices
    return results[:,1:]
#endregion

#region one_to_one_matching

def one_to_one_matching(lista, listb, threshold=20):
    list_a=[item if item!='missing' else '' for item in lista]
    list_b=[item if item!='missing' else '' for item in listb]
    
    cost_matrix = np.array([
        [abs(fuzz.WRatio(a, b)-100) for b in listb]
        for a in lista])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    dic_matches = {i:j for i, j in zip(row_ind, col_ind) if cost_matrix[i][j] <= threshold}
    return dic_matches

def delete_multispace(string):
    while '  ' in string:
        string=string.replace('  ',' ')
    return string

#endregion
