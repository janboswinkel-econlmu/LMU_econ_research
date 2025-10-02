

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
import requests

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
    for i, name in enumerate(col_names):
        globals()[f'{prefix}_{name}'] = i
        
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
def split_data_into_chunks(data, col, return_index=False):
    data=data[np.argsort(data[:, col]),:]  #sort data by column to ensure correct order
    _, indices=np.unique(data[:, col], return_index=True)
    indices=indices[np.argsort(indices)]  #sort indices to ensure correct order
    all_chunks, pre_idx = [], 0
    for z in range(1,len(indices)):
        idx = indices[z]
        mini_data = data[pre_idx:idx, :]
        if return_index:
            mini_data_idx=np.arange(pre_idx, idx).tolist()  #create index for mini_data
            all_chunks.append((mini_data_idx, mini_data))
        else:
            all_chunks.append(mini_data)
        pre_idx = idx
    #add last chunk
    mini_data = data[pre_idx:, :]
    if return_index:
        mini_data_idx=np.arange(pre_idx, data.shape[0]).tolist()
        all_chunks.append((mini_data_idx, mini_data))
    else:
        all_chunks.append(mini_data)
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

#region google

def google_ocr(image_path):
    """Performs Optical Character Recognition (OCR) on an image by invoking the Vertex AI REST API.
        Returns array with columns: index, detected_word, x1, y1, x2, y2
    """

    # Securely fetch the API key from environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GCP_API_KEY environment variable must be defined.")

    # Construct the Vision API endpoint URL
    vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    # Read and encode the local image
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Define the request payload for text detection
    request_payload = {
        "requests": [
            {
                "image": {
                        "content": encoded_image
                },
                "features": [
                    {
                        "type": "TEXT_DETECTION"
                    }
                ]
            }
        ]
    }

    # Send a POST request to the Vision API
    response = requests.post(vision_api_url, json=request_payload)
    response.raise_for_status()  # Check for HTTP errors

    response_json = response.json()

    rows=[]
    for item in response_json['responses'][0]['textAnnotations'][1:]:
        vertices=item['boundingPoly']['vertices']
        all_x, all_y = [], []
        for vertex in vertices:
            all_x.append(vertex.get('x', 0))
            all_y.append(vertex.get('y', 0))
        x1, x2, y1, y2 = min(all_x), max(all_x), min(all_y), max(all_y)
        row=np.array([item['description'], x1, y1, x2, y2], dtype=object)
        rows.append(row)
    rows=np.vstack(rows)
    rows=np.hstack((np.arange(len(rows)).reshape(-1,1), rows))

    return rows
#endregion
########################################################################################################################

#region openai

##### OPENAI SUBMIT BATCH

def make_json_format_reasoning(prompt_array, system_prompt, effort, mod, name_batch):
    #prompt_array contains 2 cols, one for index and one for string, name_batch is file name for json file
    tasks=[]
    for i in range(len(prompt_array)):
        idx=prompt_array[i,0]
        string=prompt_array[i,1]
        task = {
            "custom_id": f"{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": mod,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": string}
                ],
                "reasoning_effort": effort,
                "verbosity": "low",
                "response_format": { "type": "json_object" }
            }}
        tasks.append(task)
    file_name = f'{core_path}\\Objects\\openai_batch_history\\deletable\\{name_batch}.jsonl' #!!!important, folder path pre_defined
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')

def make_json_format(prompt_array, system_prompt, temp, mod, name_batch):
    #prompt_array contains 2 cols, one for index and one for string, name_batch is file name for json file
    tasks=[]
    for i in range(len(prompt_array)):
        idx=prompt_array[i,0]
        string=prompt_array[i,1]
        task = {
            "custom_id": f"{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": mod,
                "temperature": temp,
                "response_format": { "type": "json_object" }, #!!!default, can comment out if needed
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": string
                    }
                ],
            }
        }
    
        tasks.append(task)
    file_name = f'{core_path}\\Objects\\openai_batch_history\\deletable\\{name_batch}.jsonl' #!!!important, folder path pre_defined
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')

def make_json_format_image(prompt_array, system_prompt, temp, mod,detail, name_batch):
    #prompt_array contains 2 cols, one for index and one for string, name_batch is file name for json file
    tasks=[]
    for i in range(len(prompt_array)):
        idx=prompt_array[i,0]
        string=prompt_array[i,1]
        image_encoded=prompt_array[i,2]
        task = {
            "custom_id": f"{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": mod,
                "temperature": temp,
                "response_format": { "type": "json_object" }, #!!!default, can comment out if needed
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                            "type": "text", 
                             "text": string
                             },
                            {
                            "type": "image_url", 
                            "image_url": {"url": image_encoded , "detail": detail},}]
                        }
                        ]
                    }
            }
        tasks.append(task)
    file_name = f'{core_path}\\Objects\\openai_batch_history\\deletable\\{name_batch}.jsonl' #!!!important, folder path pre_defined
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')

def make_json_format_pdf(prompt_array, system_prompt, temp, mod, name_batch):
    #prompt_array contains 2 cols, one for index and one for string, name_batch is file name for json file
    tasks=[]
    for i in range(len(prompt_array)):
        idx=prompt_array[i,0]
        string=prompt_array[i,1]
        pdf_encoded=prompt_array[i,2]
        filename=prompt_array[i,4]
        task = {
            "custom_id": f"{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": mod,
                "temperature": temp,
                "response_format": { "type": "json_object" }, #!!!default, can comment out if needed
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                            "type": "text", 
                             "text": string
                             },
                            {
                            "type": "file",
                            "file": {
                                "filename": filename,
                                "file_data": f"data:application/pdf;base64,{pdf_encoded}"}}]
                        }
                        ]
                    }
            }
        tasks.append(task)
    file_name = f'{core_path}\\Objects\\openai_batch_history\\deletable\\{name_batch}.jsonl' #!!!important, folder path pre_defined
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')           
            
def create_json_file_openai(client, name_batch):
    batch_file = client.files.create(
      file=open(f'{core_path}\\Objects\\\\openai_batch_history\\deletable\\{name_batch}.jsonl', "rb"),
      purpose="batch"
    )
    return(batch_file)

def submit_json_file_openai(client, batch_file):
    batch_job = client.batches.create(
      input_file_id=batch_file.id,
      endpoint="/v1/chat/completions",
      completion_window="24h"
    )
    return(batch_job)

def submit_batch_job(gpt_key, prompt_array, system_prompt, temp, mod, name_batch, file_type=False, image_detail=None):
    if file_type=='text':
        if mod in ['gpt-5', "gpt-5-mini", "gpt-5-nano"]:
            make_json_format_reasoning(prompt_array, system_prompt, temp, mod, name_batch)
        else:
            make_json_format(prompt_array, system_prompt, temp, mod, name_batch)
    elif file_type=='pdf':
        make_json_format_pdf(prompt_array, system_prompt, temp, mod, name_batch)
    elif file_type=='image':
        if image_detail is None:
            raise ValueError("Specify detail for image processing, e.g. 'high', 'auto', 'low'")
        make_json_format_image(prompt_array, system_prompt, temp, mod, image_detail, name_batch)
    else:
        raise ValueError("file_type must be 'text', 'pdf', or 'image'")
    client = OpenAI(api_key=gpt_key)
    batch_file = create_json_file_openai(client, name_batch)
    batch_job = submit_json_file_openai(client, batch_file)
    return(batch_job.id)

def submit_requests(gpt_key, prompt_array, system_prompt, temp, mod, name_batch, file_type=False, image_detail=None):
    #check name of project
    filenames=get_all_filenames(f'{core_path}\\Objects\\openai_batch_history\\find_batch_ids')
    if f'{name_batch}.pkl' in filenames:
        raise ValueError(f"Batch with name {name_batch} already exists. Please choose a different name.")
    #split array into requests due to size limitations
    is_image=False if file_type=='text' else True
    requests, failed=split_into_requests(prompt_array, 125, 50000, 20, is_image) #split into batches of 1000 requests
    print(f"Number of requests: {len(requests)}", f"Number of failed requests: {len(failed)}")
    
    #if failures, raise warning
    if len(failed)>0:
        ans=input('Some requests failed, continue? y or n')
        raise ValueError("Failed requests, please check the input data.") if ans!='y' else print(f'Continuing despite {len(failed)} failed requests.')
            
    #submit each request separately and store batch_ids   
    batch_job_ids=[]
    for request in requests:
        batch_job_id= submit_batch_job(gpt_key, request, system_prompt, temp, mod, name_batch, file_type, image_detail)
        batch_job_ids.append(batch_job_id)
    
    #save batch_job_ids
    save_file(batch_job_ids, f'{core_path}\\Objects\\openai_batch_history\\find_batch_ids', name_batch)
        

def encode_image_for_json(image_path, type_img):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/{type_img};base64,{encoded_image}"

def calculate_size(image_path, system_prompt, prompt): #calculate size in mb
    if image_path is None:
        return len(prompt.encode('utf-8')) / (1024 * 1024) + len(system_prompt.encode('utf-8')) / (1024 * 1024)
    load_size = os.path.getsize(image_path)/(1024*1024)+len(prompt.encode('utf-8')) / (1024 * 1024)+ len(system_prompt.encode('utf-8')) / (1024 * 1024)
    return load_size
    
#split into either maximum number of requests or maximum size (requires last column with load size)
def split_into_requests(prompt_array, max_size, max_requests, single_max_size, image=True): #set max size to 150 in MB and requests to 50000
    requests,failed_rows = [],[]
    current_request = []
    current_size = 0
    rel_col=3 if image else 2 #if image, size is in 3rd col, if text, size is in 2nd col
    
    for row in prompt_array:
        if len(current_request) >= max_requests or current_size + row[rel_col] > max_size:
            requests.append(np.vstack(current_request))
            current_request = []
            current_size = 0
        if row[rel_col] > single_max_size: #if single row exceeds max size, skip it
            print(f"Row with index {row[0]} exceeds maximum size of {single_max_size} MB and will be skipped.")
            failed_rows.append(row)
            continue
        current_request.append(row)
        current_size += row[rel_col]
    
    if len(current_request)>0: #if not empty nor None
        requests.append(np.vstack(current_request))
    return requests, failed_rows





##### OPENAI RETRIEVE BATCH

def retrieve_and_save_data(client, batch_job_id):
    filenames= get_all_filenames(f'{core_path}\\Objects\\openai_batch_history\\complete_jobs')
    if f'{batch_job_id}.jsonl' in filenames:
        print(f"Batch {batch_job_id} already exists in complete_jobs. Rewriting it")
    
    batch_job_complete=client.batches.retrieve(batch_job_id)
    
    result_file_id = batch_job_complete.output_file_id
    result=client.files.content(result_file_id).content

    result_file_name = f'{core_path}\\Objects\\openai_batch_history\\complete_jobs\\{batch_job_id}.jsonl'
    
    with open(result_file_name, 'wb') as file:
        file.write(result) #convert into json file

def read_results(batch_job_id):
    result_file_name = f'{core_path}\\Objects\\openai_batch_history\\complete_jobs\\{batch_job_id}.jsonl'
    results = []
    with open(result_file_name, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(json_object)
    rows=np.empty((0,2),dtype=object)
    for res in results:
        task_id_string= res['custom_id']
        string_result=res['response']['body']['choices'][0]['message']['content']
        row=np.array([task_id_string, string_result],dtype=object)
        rows=np.vstack((rows, row))
    return(rows) 

def retrieve_batch_job(gpt_key, batch_job_id):
    client = OpenAI(api_key=gpt_key)
    retrieve_and_save_data(client, batch_job_id)
    results=read_results(batch_job_id)
    return(results)

def retrieve_saved_batch_job(batch_job_id):
    results = []
    with open(f'{core_path}\\Objects\\openai_batch_history\\complete_jobs\\{batch_job_id}.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    results=read_results(batch_job_id)
    return results

def retrieve_requests(gpt_key, name_batch):
    batch_job_ids=open_file(f'{core_path}\\Objects\\openai_batch_history\\find_batch_ids', name_batch)
    all_results=[]
    for batch_job_id in batch_job_ids:
        results= retrieve_batch_job(gpt_key, batch_job_id)
        all_results.append(results)
    return np.vstack(all_results)

def retrieve_saved_requests(name_batch):
    batch_job_ids=open_file(f'{core_path}\\Objects\\openai_batch_history\\find_batch_ids', name_batch)
    all_results=[]
    for batch_job_id in batch_job_ids:
        results= retrieve_saved_batch_job(batch_job_id)
        all_results.append(results)
    return np.vstack(all_results)

def extract_json(result_array, variables):
    new_array,z=result_array[:,0].reshape(-1,1),1
    new_array = new_array.astype(object)
    for var in variables:
        new_array = np.hstack((new_array, np.full((new_array.shape[0], 1), np.nan)))
        for i in range(len(new_array)):
            json_string = result_array[i, 1]
            try:
                json_data = json.loads(json_string)
                relevant_answer = json_data[var]
                new_array[i,z]=relevant_answer
            except:
                print(f'Error parsing JSON for variable {var} index {i}')
        z+=1
    return(new_array)

def extract_json_wide(result_array):
    rows=[]
    for i in range(len(result_array)):
        id=result_array[i,0]
        json_string = result_array[i, 1]
        json_data = json.loads(json_string)
        row = [id, json_string]
        for key, value in json_data.items():
            row.append(value)
        row=np.array(row, dtype=object)
        rows.append(row)
    max_length = max(len(row) for row in rows)
    for i, row in enumerate(rows):
        if len(row) < max_length:
            rows[i] = np.pad(row, (0, max_length - len(row)), constant_values=np.nan)
    new_array = np.vstack(rows)
    return(new_array)



def extract_json_custom(result_array):
    new_array,z=result_array[:,0].reshape(-1,1),0
    new_array = np.hstack((new_array, np.full((new_array.shape[0], 1), np.nan)))
    for i in range(len(new_array)):
        json_string = result_array[i, 1]
        json_data = json.loads(json_string)
        values = []
        for key, value in json_data.items():
            values.append(value)
        new_array[i,1]=values
    return(new_array)

#where the keys of the json object determine the columns
def extract_json_custom_long(result_array):
    all_rows=[]
    for i in range(len(result_array)):
        code=result_array[i,0]
        try:
            json_string = result_array[i, 1]
            json_data = json.loads(json_string)
            rows=[]
            for key, value in json_data.items():
                row=np.array([code, key, value],dtype=object)
                rows.append(row)
            rows = np.vstack(rows)
            rows = rows.reshape(-1, 3)
            all_rows.append(rows)
        except:
            row=np.array([code, np.nan, np.nan],dtype=object).reshape(-1, 3)
            all_rows.append(row)
            print(f'Error parsing JSON for index {i}')
    return(np.vstack(all_rows))

def custom_nested_keys(data,keys):
    last_orig_idx= data.shape[1]-1
    data=add_col(data, np.nan, len(keys))
    for i in range(len(data)):
        try:
            json_string=data[i,last_orig_idx]
            py_dict = ast.literal_eval(str(json_string))
            json_string = json.dumps(py_dict)
            json_data= json.loads(json_string)
            for z in range(len(keys)):
                var=keys[z]
                relevant_answer = json_data[var]
                data[i, last_orig_idx+z+1]=relevant_answer
        except:
            print(f'Error parsing JSON for index {i} with keys {keys}')
            continue
    return data

def custom_nested_keys_long(data):
    last_orig_idx= data.shape[1]-1
    all_new_rows=[]
    for i in range(len(data)):
        try:
            rest_of_row=data[i,0:last_orig_idx+1]  # get the rest of the row
            json_string=data[i,last_orig_idx]
            py_dict = ast.literal_eval(str(json_string))
            json_string = json.dumps(py_dict)
            # fixed_string = fix_double_quotes_without_colon(json_string)
            json_data= json.loads(json_string)
            rows=[]
            for key, value in json_data.items():
                row=np.array([key, value],dtype=object)
                rows.append(row)
            rows=np.vstack(rows)  # combine all rows into one array
            new_rows=np.tile(rest_of_row, (rows.shape[0], 1))  # repeat the rest of the row for each new row
            new_rows= np.hstack((new_rows, rows))  # combine the rest of the row with the new rows
            all_new_rows.append(new_rows)  # append new rows to the lis
        except:
            rest_of_row=data[i,0:last_orig_idx+1]  # get the rest of the row
            failed_rows=add_col(rest_of_row.reshape(-1,3), np.nan, 2)
            all_new_rows.append(failed_rows.reshape(-1,5))
            print(f'row json failed {i}')
            continue
    all_new_rows=np.vstack(all_new_rows)  # combine all new rows into one array
    return all_new_rows

### OPENAI instantaneous prompt
def ask_image(image_path, API_key, system_prompt, prompt, temp, mod, detail_image, img_type):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    client = OpenAI(api_key=API_key)
    image_type = f"image/{img_type}" 
    question1 = client.chat.completions.create(
        temperature=temp,
        model=mod,
        response_format={ "type": "json_object" },
        messages=[{
                    "role": "system",
                    "content": system_prompt
                },
                  {"role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_type};base64,{encoded_image}", "detail": detail_image}}]}])
    response = question1.choices[0].message.content
    return response

def ask_pdf(source, API_key, system_prompt, prompt, temp, mod):
    with open(source, "rb") as f:
        data = f.read()
    encoded_pdf = base64.b64encode(data).decode("utf-8")
    filename=source.split('\\')[-1] #get filename from path
    client = OpenAI(api_key=API_key)
    question1 = client.chat.completions.create(
        temperature=temp,
        model=mod,
        response_format={ "type": "json_object" },
        messages=[{
                    "role": "system",
                    "content": system_prompt
                },
                  {"role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": f"data:application/pdf;base64,{encoded_pdf}"}
                    }
                    ]}])
    response = question1.choices[0].message.content
    return response



def ask_multi_image(sources, API_key, system_prompt, prompt, temp, mod):
    img_type, encoded_images="image/jpeg",[]
    for source in sources:
        with open(source, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        encoded_images.append(encoded_image)
    client = OpenAI(api_key=API_key)
    messages_list=[{"role": "system", "content": system_prompt}]
    for encoded_image in encoded_images:
        messages_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img_type};base64,{encoded_image}", "detail": "high"}})
    question1 = client.chat.completions.create(
        temperature=temp,
        model=mod,
        response_format={ "type": "json_object" },
        messages=messages_list)
    response = question1.choices[0].message.content
    return response

def ask_prompt (API_key, system_prompt, prompt, temp, mod): #response=
    client = OpenAI(api_key=API_key)
    question2 = client.chat.completions.create(
        temperature=temp,
        model=mod,
        response_format={ "type": "json_object" },
        messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ],
                }
        ]
    )
    response2 = question2.choices[0].message.content
    return response2

def ask_prompt_reasoning (API_key, system_prompt, prompt, effort, mod): #response=

    client = OpenAI(api_key=API_key)
    response = client.responses.create(
        model=mod,
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        reasoning={
            "effort": effort 
        }
    )
    return response.output_text



#endregion
########################################################################################################################