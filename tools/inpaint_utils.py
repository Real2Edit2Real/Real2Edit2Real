import os
# Install Ark SDK via: pip install 'volcengine-python-sdk[ark]'
from volcenginesdkarkruntime import Ark

import base64
import numpy as np
import cv2

def b64_to_cv2_image(b64_data, save_path=None, image_shape=None):
    """
    Convert Base64 encoded image data to an OpenCV formatted numpy.array.
    
    Args:
        b64_data (str): Base64 encoded string.
        save_path (str, optional): If a path is provided, save the image to that path.
        image_shape: (w, h)
        
    Returns:
        image (np.ndarray): OpenCV formatted image (H, W, C).
    """
    try:
        # Base64 -> Binary
        img_bytes = base64.b64decode(b64_data)
        
        # Binary -> numpy.array
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        
        # numpy.array -> OpenCV image
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Decoding failed, please check if b64_data is correct")
        
        if image_shape is not None:
            image = cv2.resize(image, image_shape)
        # If saving is needed
        if save_path is not None:
            cv2.imwrite(save_path, image)
        
        return image

    except Exception as e:
        print(f"[ERROR] Unable to parse Base64 image: {e}")
        return None
    
def cv2_image_to_b64(image=None, image_path=None, image_format="png"):
    """
    Convert an OpenCV image or a local image to a standard Base64 encoded string.
    Format: data:image/<image_format>;base64,<base64_image>
    
    Args:
        image (np.ndarray, optional): OpenCV image (H, W, C), prioritized if provided.
        image_path (str, optional): Path to the image, used if image=None.
        image_format (str, optional): Image format, default is "png", must be lowercase.
        
        
    Returns:
        str: Standard Base64 encoded string.
    """
    # 1. If a path is provided, read the image
    if image is None:
        if image_path is None:
            raise ValueError("Either image or image_path must be provided")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        # If the user didn't specify a format, use the file extension
        if image_format == "png" and "." in image_path:
            ext = os.path.splitext(image_path)[1].lower().replace(".", "")
            image_format = ext if ext in ["png", "jpg", "jpeg"] else "png"

    # 2. Encode the OpenCV image into the specified format
    success, buffer = cv2.imencode(f".{image_format}", image)
    if not success:
        raise ValueError(f"Image encoding failed for format: {image_format}")

    # 3. Convert to Base64 string
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # 4. Concatenate into standard format
    return f"data:image/{image_format};base64,{img_base64}"

def initialize_client(ark_api_key):
    os.environ["ARK_API_KEY"] = ark_api_key
    client = Ark(
        # This is the default endpoint, can be configured according to your region
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        # Get your API Key from environment variables. This is the default way and can be modified as needed.
        api_key=os.environ.get("ARK_API_KEY"),
    )
    return client

def edit_image_list(client, image_path_list, prompt, save_dir, basename=None, image_shape=None):
    for image_idx, image_path in enumerate(image_path_list):
        imagesResponse = client.images.generate(
            model="doubao-seededit-3-0-i2i-250628",
            prompt=prompt,
            image=cv2_image_to_b64(image_path=image_path),
            seed=123,
            guidance_scale=5.5,
            size="adaptive",
            watermark=False,
            response_format="b64_json"
        )
        base_name = os.path.basename(image_path) if basename is None else f'{basename}_' + os.path.basename(image_path) 
        if isinstance(save_dir, list):
            save_path = os.path.join(save_dir[image_idx], f"{base_name}")
        else:
            save_path = os.path.join(save_dir, f"{base_name}")
        b64_to_cv2_image(imagesResponse.data[0].b64_json, save_path, image_shape)
        print(f"Saved edited image to: {save_path}")