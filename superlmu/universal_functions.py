

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 10:22:50 2025

@author: jan.boswinkel
"""

#region libraries
import os, pickle, random, math, json, base64, ast
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from PIL import ImageDraw
from openai import OpenAI
import inspect

#endregion
########################################################################################################################

#region data statistics
def summ(col, name_var='variable'):
    nas_count, total_count=len(np.where(pd.isna(col))[0]), len(col)
    variable_data = col[~pd.isna(col)]
    data_type = {type(item) for item in variable_data}
    if len(data_type)>1:
        summary={
            'variable': name_var,
                'n': total_count,
                 'n_isna': nas_count,
                 'data_type': 'mixed'}
        return(summary)
    if len(data_type)==0:
        summary={
            'variable': name_var,
                'n': total_count,
                 'n_isna': nas_count,
                 'data_type': 'no data type'}
        return(summary)
    #if is string, get number of unique values and print 5 most popular
    if str in data_type:
        unique_values, counts = np.unique(variable_data, return_counts=True)
        most_common = unique_values[np.argsort(counts)[-5:]]
        summary={
            'variable': name_var,
                'n': total_count,
                 'n_isna': nas_count,
                 'data_type': 'string',
                 'n_uniques': len(unique_values),
                 'most_common': most_common}
    elif all(np.issubdtype(type(item), np.number) for item in variable_data):
        summary = {
            'variable': name_var,
            'n': total_count,
            'n_isna': nas_count,
            'data_type': 'numeric (float or int)',
            'mean': np.mean(variable_data),
            'median': np.median(variable_data),
            'min': np.min(variable_data),
            'max': np.max(variable_data),
            'std_dev': np.std(variable_data)
        }
    elif any(t in data_type for t in (list, tuple, np.ndarray)):
        as_tuples = np.array([tuple(lst) for lst in variable_data], dtype=object)
        as_tuples_str = np.array([tuple(str(x) for x in tup) for tup in as_tuples], dtype=object) # Convert all elements in tuples to strings to avoid comparison errors
        unique_values, counts = np.unique(as_tuples_str, return_counts=True)
        most_common = unique_values[np.argsort(counts)[-5:]]
        summary = {
            'variable': name_var,
            'n': total_count,
            'n_isna': nas_count,
            'data_type': 'list (list, tuple or array)',
            'n_uniques': len(unique_values),
            'example': most_common}
    return(summary)
    # summary = {
    #     'data_type': data_type,
    #     'total_count':total_count,
    #     'NAs_count': nas_count,
    #     "mean": np.mean(variable_data),
    #     "median": np.median(variable_data),
    #     "min": np.min(variable_data),
    #     "max": np.max(variable_data),
    #     "percentiles": {
    #         "10th": np.percentile(variable_data, 10),
    #         "25th": np.percentile(variable_data, 25),
    #         "50th": np.percentile(variable_data, 50),
    #         "75th": np.percentile(variable_data, 75),
    #         "90th": np.percentile(variable_data, 90),
    #     },
    # }
    # return summary

#endregion
########################################################################################################################

#region data manipulation

#to show image
def show_img(path):
    img = PILImage.open(path)
    img.show()
    
#Baby function: get a given path name to a folder and return list of all the filenames contained in folder
def get_all_filenames(path, format=None, crop=False):
    if format is None:
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    elif crop:
        return [f.split('.')[0] for f in os.listdir(path) if f.endswith(format) and os.path.isfile(os.path.join(path, f))]
    else:
        return [f for f in os.listdir(path) if f.endswith(format) and os.path.isfile(os.path.join(path, f))]

def open_file (path, filename):
    with open(f'{path}\\{filename}.pkl', 'rb') as file:
        data=pickle.load(file)
    return(data)

def save_file (data, path, filename):
    with open(f'{path}\\{filename}.pkl', 'wb') as file:
        pickle.dump(data, file)

#loop to predict objects in images
def open_files_with_name (path, obj_str):
    files = get_all_filenames(path)
    filtered_files = [f for f in files if f'{obj_str}' in f]
    with open(f'{path}\\{filtered_files[0]}', 'rb') as currentfile:
        obj=pickle.load(currentfile)
    if len(filtered_files)>1:
        for file in filtered_files[1:]:
            with open(f'{path}\\{file}', 'rb') as currentfile:
                current_object=pickle.load(currentfile)
            obj=np.vstack((obj, current_object))
    return(obj)

        
#loop to predict objects in images
def open_multiple_files (path, obj_str):
    files = get_all_filenames(path)
    filtered_files = [f for f in files if f'{obj_str}' in f]
    initial_obj=open_file(path, filtered_files[0].replace('.pkl', '')) #remove .pkl from filename
    if len(filtered_files)>1:
        for file in filtered_files[1:]:
            file=file.replace('.pkl', '') #remove .pkl from filename
            current_object=open_file(path, file)
            initial_obj=np.vstack((initial_obj, current_object))
    return(initial_obj)
             
#how to make uniques preserving order in array
def uniques_preserve_order(array):
    seen = set()
    return [x for x in array if not (x in seen or seen.add(x))]

#combine multiple columns into one string column
def combine_into_string(data, list_col_indices, write_col=None):
    combined=data[:, list_col_indices[0]].astype(str)
    for i in range(1, len(list_col_indices)):
        combined = np.char.add(combined, '_')
        combined = np.char.add(combined, data[:, list_col_indices[i]].astype(str))
    if write_col is None:
        return(combined)
    elif write_col=='new':
        data=np.hstack((data, np.full((data.shape[0], 1), np.nan)))
        data[:, -1] = combined
        return(data)
    elif isinstance(write_col, int):
        data[:, write_col] = combined
        return(data)

#make unique values in a column, and count how many times they occur (sorted by frequency)
def uniques_with_counts(array):
    array= np.array(array) 
    no_nas_idx= np.where(~pd.isna(array))[0]  #get indices without NAs
    n_nas=len(array)-len(no_nas_idx)  #get number of NAs
    array= array[no_nas_idx]  #select only non-NA values
    unique, counts = np.unique(array, return_counts=True)
    ordered_array=np.column_stack((unique, counts))
    ordered_array=ordered_array.astype(object)
    ordered_array[:,1]=ordered_array[:,1].astype(int)
    ordered_array=ordered_array[np.argsort(ordered_array[:, 1])[::-1], :]
    print('number of NAs', n_nas)
    return ordered_array

#make unique values in a column, and count how many times they occur (sorted by frequency)
def uniques_no_nas(array):
    na_idx=np.where(pd.isna(array))[0]
    non_na_idx= np.setdiff1d(np.arange(0, len(array)), na_idx)  #get all indices except the ones with NAs
    unique=np.unique(array[non_na_idx])  #get unique values from non-NA indices
    print(f"Number of NAs: {len(na_idx)} out of {len(array)}")
    return unique

#merge matrices using pd DataFrame and converting back to numpy
def numpy_merge(a, b, col_ids, method, delete_key=False):
    df_a=pd.DataFrame(a)
    df_b=pd.DataFrame(b)
    #change relevant col_idx to name 'key'
    df_a.rename(columns={df_a.columns[col_ids[0]]: 'key'}, inplace=True)
    df_b.rename(columns={df_b.columns[col_ids[1]]: 'key'}, inplace=True)
    merged = pd.merge(df_a, df_b, how=method, on='key')#method can be:inner (only intersection), left, right, outer(all) 
    merged=np.array(merged)
    if delete_key:
        merged=delete_col(merged, [col_ids[0]])
    return merged

#define column index for each column name in a list
def define_col_idx(prefix, col_names):
    frame = inspect.currentframe().f_back
    for i, name in enumerate(col_names):
        frame.f_globals[f'{prefix}_{name}'] = i
        
#print random items from list or random rows from array
def random_items(data, n_elements):
    data=np.array(data)
    random_indices = random.sample(range(len(data)), n_elements)
    if len(data.shape)==1:
        return(data[random_indices])
    else:
        return(data[random_indices,:])

#reorder numpy cols (leave cols before min of col_order untouched, then implement order, then rest of cols in original order)
def reorder_cols(data, col_order):
    minimum_col = min(col_order)
    pre_cols= np.arange(0, minimum_col)
    remain_post_cols= np.arange(minimum_col+1, data.shape[1])
    remain_post_cols=[item for item in remain_post_cols if item not in col_order]
    new_data = np.hstack((data[:,pre_cols], data[:,col_order], data[:,remain_post_cols]))
    return new_data

#add column to numpy array and option to decide its position
def add_col(data, newcol, ncols, position=None):
    #if data is 1D, make it 2D
    if len(data.shape)==1:
        data = data.reshape(-1, 1)
    #if newcol is int, float, or str, make a column repeating it
    if isinstance(newcol, (int, float, str)):
        newcol = np.full((data.shape[0], ncols), newcol)
    else: #repeat array
        if len(newcol.shape) == 1:  # if newcol is 1D
            newcol = np.tile(newcol.reshape(-1,1), (1, ncols))
        else:  # if newcol is already 2D
            newcol= np.tile(newcol, (1, ncols))  # repeat each column ncols times
    #add col and attach newcol
    new_data = np.hstack((data, newcol))
    #position if specified
    
    if position is None:
        return(new_data)
    elif position==0:
        positions=[]
        for i in range(ncols):
            item= new_data.shape[1]-1-i
            positions.append(item)
        positions.append(0)
    else:
        positions=[position-1] 
        for i in range(ncols):
            item= new_data.shape[1]-1-i
            positions.append(item)
    new_data= reorder_cols(new_data, positions)
    return(new_data)

def delete_col(data, cols):
    all_cols=np.arange(0, data.shape[1], 1)
    new_cols=np.setdiff1d(all_cols, cols)  #get all columns except the ones to delete
    new_data=data[:, new_cols]  #select only the columns to keep
    return(new_data)

#split into batches (usually for parallelization)
def split_into_batches(list_array, breaks,type,return_indices=False):
    if type=='n_batches':
        batch_size= math.ceil(len(list_array) / breaks)
    elif type=='every_n':
        batch_size= breaks
    else:
        raise ValueError("type must be 'n_batches' or 'every_n'")
    if isinstance(list_array, list) or (isinstance(list_array, np.ndarray) and list_array.ndim == 1):
        batches= [list_array[i:i + batch_size] for i in range(0, len(list_array), batch_size)]
        indices= [list(range(i, min(i + batch_size, len(list_array)))) for i in range(0, len(list_array), batch_size)]
        if return_indices:
            return (batches, indices)
        else:
            return(batches)
    elif (isinstance(list_array, np.ndarray) and list_array.ndim == 2):
            batches= [list_array[i:i + batch_size, :] for i in range(0, len(list_array), batch_size)]
            indices=[list(range(i, min(i + batch_size, len(list_array)))) for i in range(0, len(list_array), batch_size)]
            if return_indices:
                return (batches, indices)
            else:
                return(batches)
    
#convert cols to specified type (str, int, float, etc)
def convert_cols_to_type(array, cols, ctype):
    array= array.astype(object)  # convert to object type to allow mixed types
    if cols=='all':
        cols = range(array.shape[1])
    for col in cols:
        array[:, col] = array[:, col].astype(ctype)  # convert columns to specified type
    return array

#split dataset into chunks based on unique values in a specific column
def split_data_into_chunks(data, col):
    """
    Splits data into chunks based on unique values in a specified column.
    """
    na_idx=np.where(pd.isna(data[:, col]))[0]
    non_na_idx=np.setdiff1d(np.arange(len(data)), na_idx)
    na_chunk=data[na_idx, :]
    print(f"Number of rows with NA in column {col}: {len(na_idx)}, a separate chunk was made with these values")
    
    non_nan_data=data[non_na_idx, :]
    non_nan_data=non_nan_data[np.argsort(non_nan_data[:, col]),:]  #sort non_nan_data by column to ensure correct order
    _, indices=np.unique(non_nan_data[:, col], return_index=True)
    all_chunks, pre_idx = [], 0
    for z in range(1,len(indices)):
        idx = indices[z]
        mini_data = non_nan_data[pre_idx:idx, :]
        all_chunks.append(mini_data)
        pre_idx = idx
    #add last chunk
    mini_data = non_nan_data[pre_idx:, :]
    all_chunks.append(mini_data)
    all_chunks.append(na_chunk)
    return all_chunks


#split multiple datasets into chunks
def split_multiple_data_into_chunks(list_datasets, list_idx_col, codes):
    uniques_list=[]
    for i, dataset in enumerate(list_datasets):
        col=list_idx_col[i]
        dataset=dataset[np.argsort(dataset[:, col]),:]
        uniques, indices=np.unique(dataset[:, col], return_index=True)
        uniques_with_indices = np.column_stack((uniques, indices))
        uniques_list.append([uniques_with_indices[np.argsort(uniques_with_indices[:, 1]), :], dataset])  # sort uniques by indices
    all_bundles=[]
    for code in codes:
        bundle=[]
        for i in range(len(list_datasets)):
            dataset_uniques, dataset=uniques_list[i][0], uniques_list[i][1]
            match_idx= np.where(dataset_uniques[:,0]==code)[0]
            if len(match_idx)==0:
                bundle.append([])
            elif len(match_idx)>0:
                idx_start= dataset_uniques[match_idx,1][0]
                idx_end=dataset_uniques[match_idx+1,1][0] if match_idx+1 < len(dataset_uniques) else len(dataset) #get index of code in uniques
                bundle.append(dataset[idx_start:idx_end, :])
        all_bundles.append(bundle)
    return all_bundles, uniques_list
    


#use dictionary to update values of correctiondata (harmonized value is key of dic and every key of dic has list of real values of correctiondata)
def use_dic_update(master_dic, correctiondata, col):
    for key in master_dic.keys():
        indeces=np.where(np.isin(correctiondata[:, col], master_dic[key]))[0]
        correctiondata[indeces, col]=key
    return(correctiondata)

#endregion
########################################################################################################################

#region images and bounding boxes

#get boxes in original x1,y1,x2,y2 format
def draw_boxes(path, boxes):
    image = PILImage.open(path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for box in boxes:
        bbox=tuple(map(float, box))
        draw.rectangle(bbox, outline='blue', width=2)
    image.show()
    
#endregion
########################################################################################################################
