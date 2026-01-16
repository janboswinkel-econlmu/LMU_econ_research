#region libraries
import os, base64, requests, numpy as np

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
        encoded_image= image.copy()
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
