import sys
sys.path.append('..')
from PIL import Image
from io import BytesIO
import requests
import os
from dotenv import load_dotenv
import json
import base64
import httpx
from openai import OpenAI
from ic_service.ic_api import ic_func_openAI
load_dotenv()

clip_url = os.environ.get("CLIP_URL")
open_ai_api = os.environ.get("OPENAI_API")
# ic_url = os.environ.get("IC_URL")
# genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
# google_api_key = os.environ.get("GOOGLE_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")



async def multi_image(list_img_base64, options):
    description_list = []
    scene_hashtag_list = []
    for img_base64 in list_img_base64:
        img_obj = {"base64_string": img_base64}

        async with httpx.AsyncClient() as client:            
            response = await client.post(clip_url,json=img_obj)
            if response.status_code == 200:
                clip_object = response.json()
                scence_hastags, enhance_description, _, _, _, _ = await ic_func_openAI(img_base64, clip_object["location"], options)
                # scene_hashtag_list.append(scence_hastags)
                description_list.append(enhance_description)
            else:
                print(f"Error calling image_classifier: {response.status_code}")
    return scene_hashtag_list, description_list


def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte).decode("utf-8")
    return img_base64


def convert_to_base64(list_path):
    img_base64_list = []
    for path in list_path:
        with Image.open(path) as img:
            base64_string = pil_image_to_base64(img)
        img_base64_list.append(base64_string)
    return img_base64_list
