#region libraries

import os
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from fuzzywuzzy import fuzz
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering
import hdbscan
from openai import OpenAI

#endregion

#region make distance matrix

def fuzzy_scores(rows_indices, row_names, col_names, matrix):
    results=[]
    for row_idx in rows_indices:
        item=row_names[row_idx]
        restofrow_indices=np.where(matrix[row_idx,:] == 0)[0]
        restofrow_names = [col_names[i] for i in restofrow_indices]
        fuzzy_scores=[fuzz.WRatio(item, other_item) for other_item in restofrow_names]
        result=np.array([row_idx, restofrow_indices, fuzzy_scores], dtype=object)
        results.append(result)
    return(np.vstack(results))


def multi_fuzzy_scores(rows_indices, row_lists, col_lists, matrix):
    """
    Calculates max fuzzy score for i, j in matrix where i, j are lists of strings
    """
    def max_fuzzy_score_2_lists(list1, list2):
        return max([fuzz.WRatio(item1, item2) for item1 in list1 for item2 in list2])
    results=[]
    for row_idx in rows_indices:
        list1=row_lists[row_idx]
        restofrow_indices=np.where(matrix[row_idx,:] == 0)[0]
        restofrow_lists = [col_lists[i] for i in restofrow_indices]
        fuzzy_scores=[max_fuzzy_score_2_lists(list1, list2) for list2 in restofrow_lists]
        result=np.array([row_idx, restofrow_indices, fuzzy_scores], dtype=object)
        results.append(result)
    return(np.vstack(results))

def zigzag_batches(list_array, breaks):
    increasing=list_array[np.argsort(list_array)]
    decreasing=list_array[np.argsort(list_array)][::-1]
    batch_size = math.ceil(len(list_array) / breaks)
    batches=[]
    for i in range(0, len(list_array), batch_size):#0,5,10,15,20...
        top= min(i + batch_size, len(list_array)) 
        internal_idx=range(i,top) #0,1,2,3,4 or 61,62,63,64,65...
        even_internal_idx= internal_idx[::2]  #0,2,4
        odd_internal_idx= internal_idx[1::2]  #1,3
        batch=increasing[even_internal_idx].tolist()+decreasing[odd_internal_idx].tolist()
        batches.append(batch)
    return batches

def parallel_fuzzy_scores(row_names, col_names, matrix, n_workers, multi):
    row_indices= np.arange(len(matrix))
    batches=zigzag_batches(row_indices, n_workers)
    if multi:
        results=Parallel(n_jobs=n_workers)(delayed(multi_fuzzy_scores)(batch, row_names, col_names, matrix) for batch in batches)
    else:
        results=Parallel(n_jobs=n_workers)(delayed(fuzzy_scores)(batch, row_names, col_names, matrix) for batch in batches)
    return(np.vstack(results))

def make_distance_matrix(row_names, col_names, n_workers, multi=False, filter=None): #if multi, col_names and row_names are list of lists
    #if filter, use it as matrix else make new matrix with 0s in upper diagonal
    if filter is not None:
        matrix=filter
    else:
        matrix = np.full((len(row_names), len(col_names)), 0, dtype=float)
        np.fill_diagonal(matrix, 100)  # fill diagonal with 100
        matrix[np.tril_indices_from(matrix, k=-1)] = 100 #values below diag are 100
        
    #parallel operations
    results= parallel_fuzzy_scores(row_names, col_names, matrix, n_workers, multi)
    for result in results:
        row_idx, col_indices, fuzzy_scores = result
        matrix[row_idx, col_indices] = fuzzy_scores  #fill in the scores for the row
        
    #copy one side of diagonal into other side and convert into distances (1-x/100)
    matrix[np.tril_indices_from(matrix, k=0)] = 0  #set diagonal and everything below it to 0
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, 100)  # fill diagonal with 100
    matrix = 1 - matrix/100
    sim_matrix=1- matrix
    return matrix, sim_matrix

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