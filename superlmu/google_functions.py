#region libraries

import os
import base64
import json
import re
import mimetypes
from pathlib import Path

import requests
import numpy as np
from google import genai
from google.genai import types
from google.cloud import storage
from google.oauth2 import service_account

from .universal_functions import get_all_filenames, save_file, open_file

#endregion
########################################################################################################################

#region google

def google_ocr(image, format):
    """Performs Optical Character Recognition (OCR) on an image by invoking the Vertex AI REST API.
        Returns array with columns: index, detected_word, x1, y1, x2, y2
        Format can be "path" or "encoded"
    """

    # Securely fetch the API key from environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable must be defined.")

    # Construct the Vision API endpoint URL
    vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    # Read and encode the local image
    if format=='path':
        with open(image, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
    elif format=='encoded':
        encoded_image= image
    else:
        print("format must be 'path' or 'encoded'")

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

#region google single prompt functions
def count_tokens(client, mod, system_prompt, prompt, image_path=None):
    contents_ = []

    if system_prompt:
        contents_.append(f"System: {system_prompt}")

    if image_path is not None:
        image_path = Path(image_path)
        image_bytes = image_path.read_bytes()
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"

        contents_.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )
        )

    contents_.append(prompt)

    response = client.models.count_tokens(
        model=mod,
        contents=contents_
    )
    return response.total_tokens

def ask_gemini(mod, system_prompt, prompt, image_path, temp, thinking_param, SCHEMA=None, highresolution=False):
    """
    Generate content using the Gemini model with optional image input.
    Args:
        model (str): The name of the Gemini model to use.
        system_prompt (string): The main system instructions for content generation.
        prompt (str): The flexible prompt for content generation.
        image_path (str or None): The path to an image file to upload, or None if no image is used.
        temperature (float): The temperature setting for content generation.
        thinking_param (str): The thinking configuration level. "low", "medium", or "high". if 3 else, from 0 to 24576 for 2.5 flash, 32768 for 2.5 pro)
    """
    my_api_key=os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=my_api_key)
    if image_path is not None:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        mime_type, _ = mimetypes.guess_type(image_path)
        contents_list=[{"role": "user", "parts": [{"text": prompt}, {"inline_data": {"mime_type": mime_type, "data": img_bytes}}]}]
    else:
        contents_list=[{"role": "user", "parts": [{"text": prompt}]}]

    if '3' in mod:
        if thinking_param not in ['minimal','low', 'medium', 'high']:
            raise ValueError("For Gemini 3 models, thinking_param must be 'minimal','low', 'medium', or 'high'.")
        thinking_config = types.ThinkingConfig(thinkingLevel=thinking_param)
    elif '2.5' in mod:
        if not isinstance(thinking_param, int):
            raise ValueError("For Gemini 2.5 models, thinking_param must be int (up to 24576 for flash, 32768 for pro).")
        thinking_config = types.ThinkingConfig(thinking_budget=thinking_param)
    else:
        thinking_config = None

    response = client.models.generate_content(
        model=mod, contents=contents_list,
        config=types.GenerateContentConfig(
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH if highresolution else None,
            system_instruction=system_prompt,
            thinking_config=thinking_config,
            temperature=temp,
            response_mime_type='application/json',
            response_json_schema=SCHEMA if SCHEMA is not None else None #! comment out .model_json_schema() 
        ),
    )

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    cached_content_tokens = int(getattr(usage, "cached_content_token_count", 0) or 0)
    tool_use_prompt_tokens = int(getattr(usage, "tool_use_prompt_token_count", 0) or 0)
    candidates_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    reasoning_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)

    inputs_tokens= prompt_tokens + cached_content_tokens + tool_use_prompt_tokens
    output_tokens=candidates_tokens

    return response.text, inputs_tokens, output_tokens+reasoning_tokens

def list_gemini_files():
    my_api_key=os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=my_api_key)
    files = client.files.list()
    allfiles=[]
    for file in files:
        print(file.name, file.create_time)
        allfiles.append(file.name)
    return allfiles

def delete_gemini_files(files_list):
    my_api_key=os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=my_api_key)
    for file_name in files_list:
        client.files.delete(name=file_name)
        print('Deleted file:', file_name)



#array contains key, system prompt, prompt, image_path, size

def write_batch_gemini(mod, array, temp, thinking_param, path_to_gemini_jobs, request_name, SCHEMA=None, highresolution=False):

    """
    array needs to be a numpy object with 5 columns containing:
    key, system prompt, prompt, encoded img (encoding;;mime_type;;img_ref), size
    encoded img must have encoding, mime type and b64 data (or gcs reference) separated by ';;' if not None
    """
    #make sure request does not exist
    allfiles=get_all_filenames(f'{path_to_gemini_jobs}\\input_json')
    relevant_files = [re.sub(r'\d+$', '', file.rsplit('.', 1)[0]) for file in allfiles]
    if request_name in relevant_files:
        raise ValueError("A request with this name already exists. Please choose a different request_name.")

    #safety checks for thinking params
    if '3' in mod:
            if thinking_param not in ['minimal','low', 'medium', 'high']:
                raise ValueError("For Gemini 3 models, thinking_param must be 'minimal','low', 'medium', or 'high'.")
            thinking_param=thinking_param.upper()
            thinking_config={"includeThoughts": True, "thinkingLevel": thinking_param}
    elif '2.5' in mod:
        if not isinstance(thinking_param, int):
            raise ValueError("For Gemini 2.5 models, thinking_param must be int (up to 24576 for flash, 32768 for pro).")
        thinking_config = {"thinking_budget": thinking_param}
    else:
        thinking_config = None
    if temp<0 or temp>2:
        raise ValueError("Temperature must be between 0 and 2.")

    out_path = os.path.join(f'{path_to_gemini_jobs}\\input_json', f"{request_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(array):
            key, system_prompt, prompt, img_ref, _= item

            if img_ref is not None:
                encoding, mime_type, ref=img_ref.split(';;')
                if encoding=='gcs':
                    contents_list=[{"role": "user", "parts": [{"text": prompt}, {"fileData": {"mimeType": mime_type, "fileUri": ref}}]}]
                elif encoding=='b64':
                    contents_list=[{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": mime_type, "data": ref}}]}]
                else:
                    raise ValueError("Invalid encoding type. Must be 'gcs' or 'b64'.")
            else:
                contents_list=[{"role": "user", "parts": [{"text": prompt}]}]

            req = {
                "custom_id": key,
                "request": { 
                    "contents": contents_list, 
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}]
                    },
                    "generationConfig": { 
                        "temperature": temp,
                        "thinkingConfig": thinking_config,
                        "responseMimeType": "application/json",
                        "responseJsonSchema": SCHEMA,#.model_json_schema() if SCHEMA is not None else None
                        "mediaResolution": types.MediaResolution.MEDIA_RESOLUTION_HIGH if highresolution else None
                    } 
                }
            }
            f.write(json.dumps(req) + "\n")

def submit_batch_gemini(mod, path_to_gemini_jobs, request_name):
    all_files=get_all_filenames(f'{path_to_gemini_jobs}\\input_json')
    #delete the .extensions and numbers
    relevant_files = [re.sub(r'\d+$', '', file.rsplit('.', 1)[0]) for file in all_files]
    indices=[i for i, item in enumerate(relevant_files) if item==request_name]
    print('Files to process:', [all_files[i] for i in indices])
    if len(relevant_files)==0:
        raise ValueError("No files found matching the request_name.")
    relevant_files=[all_files[i] for i in indices]
    batch_names=[file.rsplit('.', 1)[0] for file in relevant_files]

    my_api_key=os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=my_api_key)

    #upload file, submit it to batch, delete file
    for i in range(len(relevant_files)):
        uploaded_file = client.files.upload(
                file=f'{path_to_gemini_jobs}\\input_json\\{relevant_files[i]}',
                config=types.UploadFileConfig(display_name=batch_names[i], mime_type='jsonl')
            )
        file_name = uploaded_file.name
        uploaded_file = client.files.get(name=file_name)
        print('Uploaded file:', uploaded_file.name)

        file_batch_job = client.batches.create(
        model=mod,
        src=uploaded_file.name,
        config={
            'display_name': batch_names[i],
        },
        )
        print(f"Created batch job: {file_batch_job.name}")
        save_file(file_batch_job.name, f'{path_to_gemini_jobs}\\links', f"{batch_names[i]}")
        client.files.delete(name=uploaded_file.name)

def monitor_status(show_num_jobs=10):
    """
    Lists the last batch jobs. Fetches full details for each to ensure 
    request counts are accurate.
    """
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    list_response = client.batches.list(config={'page_size': show_num_jobs})
    count = 0
    for job in list_response:
        if count >= show_num_jobs:
            break
        name=job.name
        status = str(job.state).replace("JobState.JOB_STATE_", "")
        display_name = getattr(job, 'display_name', '') or "N/A"
        if len(display_name) > 18:
            display_name = display_name[:15] + "..."
        print(f" {name}, {display_name:<20} | {status:<12}")
        count += 1

def retrieve_batch_gemini(request_name, path_to_gemini_jobs):
    my_api_key=os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=my_api_key)
    allfiles=get_all_filenames(f'{path_to_gemini_jobs}\\links')
    relevant_files = [re.sub(r'\d+$', '', file.rsplit('.', 1)[0]) for file in allfiles]
    indices=[i for i, item in enumerate(relevant_files) if item==request_name]
    relevant_files=[allfiles[i] for i in indices]
    print('Files to process:', relevant_files)
    all_rows=[]
    for file in relevant_files:
        filename=file.rsplit('.', 1)[0]
        batch_id=open_file(f'{path_to_gemini_jobs}\\links', f'{filename}')
        batch_job = client.batches.get(name=batch_id)
        if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
            if batch_job.dest and batch_job.dest.file_name:
                # Results are in a file
                result_file_name = batch_job.dest.file_name
                print(f"Results are in file: {result_file_name}")

                print("Downloading result file content...")
                file_content = client.files.download(file=result_file_name)
                #extract responses in lists of jsons
                res=(file_content.decode('utf-8'))
                lines = res.strip().splitlines()
                parsed_data = []
                for line in lines:
                    parsed_data.append(json.loads(line))
            rows=[]
            for dic in parsed_data:
                try:
                    response= dic['response']['candidates'][0]['content']['parts'][-1]['text'] #! modified to ignore thoughts
                    input_tokens=  dic['response']['usageMetadata']['promptTokenCount']
                    output_tokens=dic['response']['usageMetadata']['candidatesTokenCount']
                    if 'thoughtsTokenCount' in dic['response']['usageMetadata']:
                        output_tokens+=dic['response']['usageMetadata']['thoughtsTokenCount']
                    row=np.array([dic['custom_id'], response, input_tokens, output_tokens],dtype=object)
                    rows.append(row)
                except:
                    print('Error in response for custom_id:', dic['custom_id'])
            all_rows.extend(rows)
            rows=np.vstack(rows)
            save_file(rows, f'{path_to_gemini_jobs}\\output_json', f"{filename}")
            if batch_job.src and batch_job.src.file_name:
                try:
                    client.files.delete(name=batch_job.src.file_name)
                except Exception as e:
                    print(f"Warning to delete {batch_job.src.file_name}: {e}")
        else:
            print(f"Batch job is not yet complete. Current state: {batch_job.state.name}")
    all_rows=np.vstack(all_rows)
    return all_rows


def cancel_batch(gemini_path, request_name=None, specific_request_name=None):
    my_api_key=os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=my_api_key)
    if request_name:
        allfiles=get_all_filenames(f'{gemini_path}\\links')
        relevant_files = [re.sub(r'\d+$', '', file.rsplit('.', 1)[0]) for file in allfiles]
        indices=[i for i, item in enumerate(relevant_files) if item==request_name]
        relevant_files=[allfiles[i] for i in indices]
        print('Files to process:', relevant_files)
        for file in relevant_files:
            filename=file.rsplit('.', 1)[0]
            batch_id=open_file(f'{gemini_path}\\links', f'{filename}')
            client.batches.cancel(name=batch_id)
            print(f'Cancelled job: {batch_id}')
    elif specific_request_name:
        batch_id=open_file(f'{gemini_path}\\links', f'{specific_request_name}')
        client.batches.cancel(name=batch_id)
        print(f'Cancelled job: {specific_request_name}')

#endregion
########################################################################################################################

#region google cloud storage functions

def upload_file(client, bucket_name, folder_path, image_name):
    bucket = client.bucket(bucket_name)
    blob=bucket.blob(image_name)
    source_file_name = rf'{folder_path}\{image_name}'
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to gs://{bucket_name}//{blob.name}")

def upload_files(path_to_creds, bucket_name, folder_path, image_names):
    creds = service_account.Credentials.from_service_account_file(path_to_creds)
    client = storage.Client(credentials=creds, project=creds.project_id)
    bucket = client.bucket(bucket_name)
    for image_name in image_names:
        blob=bucket.blob(image_name)
        source_file_name = rf'{folder_path}\{image_name}'
        blob.upload_from_filename(source_file_name)
        print(f"Uploaded {source_file_name} to gs://{bucket_name}//{blob.name}")

#endregion
########################################################################################################################

#region google batch vertex ai functions

def submit_batch_vertex(mod, path_to_gemini_jobs, request_name, path_to_creds, gcs_bucket_name):
    """
    gcs bucket needs inputs and outputs folders
    """
    all_files = get_all_filenames(f'{path_to_gemini_jobs}\\input_json')
    relevant_files = [file.rsplit('.', 1)[0] for file in all_files]
    indices = [i for i, item in enumerate(relevant_files) if item == request_name]
    print('Files to process:', [all_files[i] for i in indices])
    
    if len(relevant_files) == 0:
        raise ValueError("No files found matching the request_name.")
        
    relevant_files = [all_files[i] for i in indices]
    batch_names = [file.rsplit('.', 1)[0] for file in relevant_files]

    creds = service_account.Credentials.from_service_account_file(path_to_creds)
    storage_client = storage.Client(credentials=creds, project=creds.project_id)
   
    bucket = storage_client.bucket(gcs_bucket_name)
    location = "global" # Or your preferred Vertex AI region
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_creds
    client = genai.Client(vertexai=True, project=creds.project_id, location=location) #http_options={"api_version": "v1beta1"}

    for i in range(len(relevant_files)):
        local_file_path = f'{path_to_gemini_jobs}\\input_json\\{relevant_files[i]}'
        
        input_blob_name = f'inputs/{relevant_files[i]}'
        blob = bucket.blob(input_blob_name)
        blob.upload_from_filename(local_file_path)
        gcs_input_uri = f"gs://{gcs_bucket_name}/{input_blob_name}"
        print('Uploaded file to GCS:', gcs_input_uri)

        gcs_output_uri = f"gs://{gcs_bucket_name}/outputs/{batch_names[i]}/"
        file_batch_job = client.batches.create(
            model=mod,
            src=gcs_input_uri,   
            config={
                "dest": gcs_output_uri,  
                "display_name": batch_names[i],
            },
        )
        print(f"Created Vertex batch job: {file_batch_job.name}")
        save_file(file_batch_job.name, f'{path_to_gemini_jobs}\\links', f"{batch_names[i]}")
        
        # --- DIFFERENCE 5: NO IMMEDIATE DELETION ---
        # DO NOT DELETE THE FILE HERE! 
        # client.files.delete(name=uploaded_file.name) is removed.
        # You must wait for the job to reach 'SUCCEEDED' or 'FAILED' state 
        # before running blob.delete()

def vertex_batch_status(path_to_creds, path_to_gemini_jobs, name_batch):
    creds = service_account.Credentials.from_service_account_file(path_to_creds)
    location = "global" # Or your preferred Vertex AI region
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_creds
    client = genai.Client(vertexai=True, project=creds.project_id, location=location)

    namebatch=open_file(f'{path_to_gemini_jobs}\\links', name_batch)
    job = client.batches.get(name=namebatch)
    print("state:", job.state)

    if getattr(job, "error", None):
        print("error code:", job.error.code)
        print("error message:", job.error.message)
        print("error details:", getattr(job.error, "details", None))

def retrieve_batch_vertex(path_to_creds, gcs_bucket_name, path_to_gemini_jobs, request_name):
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_creds
    creds = service_account.Credentials.from_service_account_file(path_to_creds)
    storage_client = storage.Client(credentials=creds, project=creds.project_id)
    location = "global" # Or your preferred Vertex AI region
    client = genai.Client(vertexai=True, project=creds.project_id, location=location)
    bucket = storage_client.bucket(gcs_bucket_name)
    
    allfiles = get_all_filenames(f'{path_to_gemini_jobs}\\links')
    relevant_files_base = [file.rsplit('.', 1)[0] for file in allfiles]
    indices = [i for i, item in enumerate(relevant_files_base) if item == request_name]
    relevant_files = [allfiles[i] for i in indices]

    print('Files to process:', relevant_files)

    all_rows = []

    for file in relevant_files:
        filename = file.rsplit('.', 1)[0]
        batch_id = open_file(f'{path_to_gemini_jobs}\\links', f'{filename}')
        batch_job = client.batches.get(name=batch_id)

        if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
            print(f"Batch succeeded for {filename}")

            # Look for result files in GCS under outputs/
            prefix = f"outputs/{filename}"
            blobs = list(bucket.list_blobs(prefix=prefix))

            if not blobs:
                print(f"No output files found in GCS for prefix: {prefix}")
                continue

            parsed_data = []

            for blob in blobs:
                print(f"Downloading from GCS: gs://db_1v/{blob.name}")
                file_content = blob.download_as_text()
                lines = file_content.strip().splitlines()

                for line in lines:
                    try:
                        parsed_data.append(json.loads(line))
                    except Exception as e:
                        print(f"Error parsing line in {blob.name}: {e}")

            rows = []
            # failed_responses=[]
            for dic in parsed_data:
                try:
                    response = dic['response']['candidates'][0]['content']['parts'][-1]['text']
                    input_tokens = dic['response']['usageMetadata']['promptTokenCount']
                    output_tokens = dic['response']['usageMetadata']['candidatesTokenCount']

                    if 'thoughtsTokenCount' in dic['response']['usageMetadata']:
                        output_tokens += dic['response']['usageMetadata']['thoughtsTokenCount']

                    row = np.array(
                        [dic['custom_id'], response, input_tokens, output_tokens],
                        dtype=object
                    )
                    rows.append(row)

                except Exception:
                    print('Error in response for custom_id:', dic.get('custom_id', 'UNKNOWN'))
                    # failed_responses.append(dic['status'])

            if rows:
                all_rows.extend(rows)
                rows = np.vstack(rows)
                save_file(rows, f'{path_to_gemini_jobs}\\output_json', f"{filename}")
            else:
                print(f"No valid rows parsed for {filename}")

            # Optional cleanup of Gemini source file
            if batch_job.src and batch_job.src.file_name:
                try:
                    client.files.delete(name=batch_job.src.file_name)
                except Exception as e:
                    print(f"Warning deleting {batch_job.src.file_name}: {e}")

        else:
            print(f"Batch job is not yet complete. Current state: {batch_job.state.name}")

    if all_rows:
        all_rows = np.vstack(all_rows)
    else:
        all_rows = np.empty((0, 4), dtype=object)

    return all_rows

#endregion
########################################################################################################################
