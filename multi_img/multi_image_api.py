from PIL import Image
from io import BytesIO
import requests
import os
from dotenv import load_dotenv
import json
import base64
import ast
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

load_dotenv()

clip_url = os.environ.get("CLIP_URL")
ic_url = os.environ.get("IC_URL")
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
# google_api_key = os.environ.get("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")


def multi_image(list_img_base64, options):
    description_list = []
    scene_hashtag_list = []
    for img_base64 in list_img_base64:
        clip_obj = {"base64_string": img_base64}
        clip_object = requests.post(clip_url, json=clip_obj)
        clip_extract = json.loads(clip_object.text)
        ic_obj = {
            "base64_string": img_base64,
            "location": clip_extract["location"],
            "options": ["problem1"]
        }
        ic_object = requests.post(ic_url, json=ic_obj)
        print(ic_object)
        ic_extract = json.loads(ic_object.text)
        ic_extract = ast.literal_eval(ic_extract["ic"])
        print(ic_extract)
        scence_hastags, enhance_description, _, _, _, _ = ic_extract
        scene_hashtag_list.append(scence_hastags)
        description_list.append(enhance_description)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"There are some description about about same location as following: \
              {description_list}. Write a detailed description about this location. \
              Note that maybe there are some duplicate information, remove it and only use available information.",
            },
        ]
    )
    gemini_pro_llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                            # google_api_key=google_api_key,
                                            temperature=0.2)
    content = gemini_pro_llm.invoke([message]).content
    return list(set(scene_hashtag_list)), content


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
