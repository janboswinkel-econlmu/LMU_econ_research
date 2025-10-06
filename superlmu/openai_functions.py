#region libraries

import os
import json
import base64
import numpy as np
import ast
from openai import OpenAI
from .universal_functions import get_all_filenames, save_file, open_file, add_col
import pandas as pd

#endregion
########################################################################################################################
########################################################################################################################
########################################################################################################################

#region background_making_json_formats
def make_json_format_reasoning(path_to_openai_logs, prompt_array, system_prompt, effort, mod, name_batch):
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
    file_name = f'{path_to_openai_logs}\\deletable\\{name_batch}.jsonl' #!!!important, folder path pre_defined
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')

def make_json_format(path_to_openai_logs, prompt_array, system_prompt, temp, mod, name_batch):
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
    file_name = f'{path_to_openai_logs}\\deletable\\{name_batch}.jsonl'
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')

def make_json_format_image(path_to_openai_logs, prompt_array, system_prompt, temp, mod,detail, name_batch):
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
    file_name =f'{path_to_openai_logs}\\deletable\\{name_batch}.jsonl' #!!!important, folder path pre_defined
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')

def make_json_format_pdf(path_to_openai_logs, prompt_array, system_prompt, temp, mod, name_batch):
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
    file_name = f'{path_to_openai_logs}\\deletable\\{name_batch}.jsonl' #!!!important, folder path pre_defined
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')           
#endregion
########################################################################################################################

#region background_submit to openai
def create_json_file_openai(path_to_openai_logs, client, name_batch):
    batch_file = client.files.create(
      file=open(f'{path_to_openai_logs}\\deletable\\{name_batch}.jsonl', "rb"),
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

def submit_batch_job(path_to_openai_logs, prompt_array, system_prompt, temp, mod, name_batch, file_type=False, image_detail=None):
    if file_type=='text':
        if mod in ['gpt-5', "gpt-5-mini", "gpt-5-nano"]:
            make_json_format_reasoning(path_to_openai_logs, prompt_array, system_prompt, temp, mod, name_batch)
        else:
            make_json_format(path_to_openai_logs, prompt_array, system_prompt, temp, mod, name_batch)
    elif file_type=='pdf':
        make_json_format_pdf(path_to_openai_logs, prompt_array, system_prompt, temp, mod, name_batch)
    elif file_type=='image':
        if image_detail is None:
            raise ValueError("Specify detail for image processing, e.g. 'high', 'auto', 'low'")
        make_json_format_image(path_to_openai_logs, prompt_array, system_prompt, temp, mod, image_detail, name_batch)
    else:
        raise ValueError("file_type must be 'text', 'pdf', or 'image'")
    client = OpenAI()
    batch_file = create_json_file_openai(path_to_openai_logs, client, name_batch)
    batch_job = submit_json_file_openai(client, batch_file)
    return(batch_job.id)

#endregion
########################################################################################################################

#region background retrieve from openai
def retrieve_and_save_data(path_to_openai_logs, client, batch_job_id):
    filenames= get_all_filenames(f'{path_to_openai_logs}\\complete_jobs')
    if f'{batch_job_id}.jsonl' in filenames:
        print(f"Batch {batch_job_id} already exists in complete_jobs. Rewriting it")
    
    batch_job_complete=client.batches.retrieve(batch_job_id)
    
    result_file_id = batch_job_complete.output_file_id
    result=client.files.content(result_file_id).content

    result_file_name = f'{path_to_openai_logs}\\complete_jobs\\{batch_job_id}.jsonl'
    
    with open(result_file_name, 'wb') as file:
        file.write(result) #convert into json file
    
    with open(result_file_name, 'wb') as file:
        file.write(result) #convert into json file

def read_results(path_to_openai_logs, batch_job_id):
    result_file_name = f'{path_to_openai_logs}\\{batch_job_id}.jsonl'
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

def retrieve_batch_job(path_to_openai_logs, batch_job_id):
    client = OpenAI()
    retrieve_and_save_data(path_to_openai_logs, client, batch_job_id)
    results=read_results(path_to_openai_logs, batch_job_id)
    return(results)

def retrieve_saved_batch_job(path_to_openai_logs, batch_job_id):
    results = []
    with open(f'{path_to_openai_logs}\\{batch_job_id}.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    results=read_results(path_to_openai_logs, batch_job_id)
    return results

def retrieve_saved_requests(name_batch):
    batch_job_ids=open_file(f'{core_path}\\Objects\\openai_batch_history\\find_batch_ids', name_batch)
    all_results=[]
    for batch_job_id in batch_job_ids:
        results= retrieve_saved_batch_job(batch_job_id)
        all_results.append(results)
    return np.vstack(all_results)
#endregion
########################################################################################################################

#region background_other

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

def create_subfolders(base_path):
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path '{base_path}' does not exist.")
    subfolders = ['find_batch_ids', 'complete_jobs', 'deletable']
    for folder in subfolders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)

#endregion
########################################################################################################################

########################################################################################################################
########################################################################################################################
########################################################################################################################

#region functions to submit and extract requests

def submit_requests(path_to_openai_logs, prompt_array, system_prompt, temp, mod, name_batch, file_type=False, image_detail=None):
    
    #ensures path_to_openai_logs exists and has required subfolders (find_batch_ids, complete_jobs, deletable)
    create_subfolders(path_to_openai_logs)

    #check name of project
    filenames=get_all_filenames(f'{path_to_openai_logs}\\find_batch_ids')
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
        batch_job_id= submit_batch_job(path_to_openai_logs, request, system_prompt, temp, mod, name_batch, file_type, image_detail)
        batch_job_ids.append(batch_job_id)
    
    #save batch_job_ids
    save_file(batch_job_ids, f'{path_to_openai_logs}\\find_batch_ids', name_batch)

def retrieve_requests(path_to_openai_logs, name_batch, already_saved=False):
    batch_job_ids=open_file(f'{path_to_openai_logs}\\find_batch_ids', name_batch)
    all_results=[]
    for batch_job_id in batch_job_ids:
        if already_saved:
            results = retrieve_saved_batch_job(path_to_openai_logs, batch_job_id)
        else:
            results = retrieve_batch_job(path_to_openai_logs, batch_job_id)
        all_results.append(results)
    return np.vstack(all_results)
#endregion
########################################################################################################################

#region functions to extract json
def extract_json(result_array, variables):
    """
    Args:
        result_array (numpy.ndarray): A 2D array where the first column contains IDs and the second column contains JSON strings.
        variables (list): A list of keys to extract from the JSON strings (from the 1st-level of keys)

    Extracts specific variables from JSON strings stored in the second column of a 2D array. Every variable is a separate column.

    Returns:
        numpy.ndarray: A new 2D array with the first column as IDs and additional columns for each extracted variable.
    """    
    last_col=result_array.shape[1]-1
    z=last_col+1
    errors1, errors2, errors3=[], [], []
    for var in variables:
        result_array = add_col(result_array, np.nan, 1)
        for i in range(len(result_array)):
            json_string = result_array[i, last_col]
            if pd.isna(json_string):
                errors1.append((i, var))
                continue
            try:
                json_data = json.loads(json.dumps(json_string)) if isinstance(json_string, dict) else json.loads(json_string)
            except:
                errors2.append((i, var))
                continue
            try:
                relevant_answer = json_data[var]
            except:
                errors3.append((i, var))
                continue
            result_array[i,z]=relevant_answer
        print(f'Number of rows where variable {var} was not extracted (NAs): {len(errors1)+len(errors2)+len(errors3)} / {len(result_array)}, where {len(errors1)} were NaNs, {len(errors2)} could not be parsed for other reasons, and {len(errors3)} could be parsed but didnt have key')
        z+=1
    return(result_array)

def extract_json_custom(result_array, output_format):
    """
    Args:
        result_array (numpy.ndarray): A 2D array where the first column contains IDs and the second column contains JSON strings.
                                      The first level of keys is not known in advance (or may change per row)
        output_format(str): 'long' or 'list'. 'long' creates one row per key, 'list' creates a list per key 
                            and returns all the keys in the JSON string as list of lists in a single row
    Extracts all the values from every key in the JSON strings stored in the second column of a 2D array. Returns every key as a separate row or as a list in a list of lists.
    
    Returns:
        numpy.ndarray: A new 2D array with the first column as IDs and additional columns for each extracted variable.
    """   
    # new_array=result_array[:,0].reshape(-1,1)
    # new_array = np.hstack((new_array, np.full((new_array.shape[0], 1), np.nan)))
    new_array=add_col(result_array, np.nan, 1)
    rows=[]
    errors, emptys=[],[]
    for i in range(len(new_array)):
        id=new_array[i,0]
        json_string = new_array[i, 1]
        try:
            json_data = json.loads(json_string)
        except:
            errors.append(i)
        values = []
        if len(json_data)==0 and output_format=='long':
            row=np.array([id, json_string, np.nan],dtype=object).reshape(-1,3)
            rows.append(row)
            emptys.append(i)
            continue
        for _, value in json_data.items():
            if output_format=='long':
                row=np.array([id, json_string, value],dtype=object).reshape(-1,3)
                rows.append(row)
            if output_format=='list':
                values.append(value)
        if output_format=='list':
            new_array[i,3]=values
    print(f'Number of rows where json could not be parsed: {len(errors)} / {len(new_array)} and number of empty jsons: {len(emptys)}')
    if output_format=='long':
        return(np.vstack(rows))
    if output_format=='list':
        return new_array
#endregion
########################################################################################################################

#region instantaneous calls

def ask_image(image_path, system_prompt, prompt, temp, mod, detail_image, img_type):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    client = OpenAI()
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

def ask_pdf(source, system_prompt, prompt, temp, mod):
    with open(source, "rb") as f:
        data = f.read()
    encoded_pdf = base64.b64encode(data).decode("utf-8")
    filename=source.split('\\')[-1] #get filename from path
    client = OpenAI()
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

def ask_multi_image(sources, system_prompt, prompt, temp, mod):
    img_type, encoded_images="image/jpeg",[]
    for source in sources:
        with open(source, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        encoded_images.append(encoded_image)
    client = OpenAI()
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

def ask_prompt (system_prompt, prompt, temp, mod): #response=
    client = OpenAI()
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

def ask_prompt_reasoning (system_prompt, prompt, effort, mod): 
    client = OpenAI()
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
