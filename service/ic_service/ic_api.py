from dotenv import load_dotenv
import os
from .image_description import extract_hashtag_gpt, image_summarizing_gpt
from .openai_detection import detect_obj_openai, combine_detect_df
import google.generativeai as genai
import base64
from PIL import Image
import requests
from io import BytesIO, StringIO
import json
import ast
import pandas as pd
import re
from openai import OpenAI
import httpx
load_dotenv()
gemini_key = os.environ.get("GEMINI_KEY")
genai.configure(api_key=gemini_key)
open_ai_api = os.environ.get("OPENAI_API_KEY")
# Load gemini-pro-vision
generation_config = {"temperature": 0.2}


def convert(base64_str):
    image_data = base64.b64decode(base64_str)
    image = BytesIO(image_data)
    img = Image.open(image)
    img.save("input.jpg")
    return img


def convert_OCR_result(df):
    # split 'text_region' has 4 points (x,y) to 4 columns (xmin, ymin, xmax, ymax)
    try:
        df['xmin'] = df['text_region'].apply(lambda x: x[0][0])
        df['ymin'] = df['text_region'].apply(lambda x: x[0][1])
        df['xmax'] = df['text_region'].apply(lambda x: x[2][0])
        df['ymax'] = df['text_region'].apply(lambda x: x[2][1])
        df = df.drop(columns=['text_region'])
        return df
    except:
        # if df is empty
        return None


def ensure_dataframe(data_str):
    if "Empty DataFrame" in data_str:
        columns_match = re.search(r'Columns: \[(.*)\]', data_str)
        columns = columns_match.group(1).split(', ') if columns_match else []
        df = pd.DataFrame(columns=columns)
        return df
    else:
        input_str = '\n'.join([line.lstrip()
                              for line in data_str.splitlines()])
        input_data = StringIO(input_str)
        df = pd.read_csv(input_data, delim_whitespace=True)
        return df




async def ic_func_openAI(base64_str, location, options):
    myobj = {'base64_string': base64_str}
    yolo_url = os.environ.get("OD_URL")
    async with httpx.AsyncClient() as client:
        x = await client.post(yolo_url, json=myobj)
        if x.status_code == 200:
            y = x.json()
            final_df = ast.literal_eval(y["info_od"].replace("nan", "None"))
            all_count_df = ast.literal_eval(y["all_count_df"].replace("nan", "None"))
            heineken_brand_count_df = ast.literal_eval(
                y["heineken_brand_count_df"].replace("nan", "None"))
            competitor_brand_count_df = ast.literal_eval(
                y["competitor_brand_count_df"].replace("nan", "None"))
            try:
                yolo_df = final_df[final_df['confidence'] > 0.7]
            except:
                yolo_df = final_df
        else:
            print(f"Error calling image_classifier: {x.status_code}")
            return None, None, None, None, None, None

    gpt = OpenAI(api_key=open_ai_api)

    enhanced_description = image_summarizing_gpt(base64_str, options, location, yolo_df, gpt)
    scene_hashtags = 'null'

    heineken_brand_count_df_1, competitor_brand_count_df_1 = detect_obj_openai(img_str, gpt)

    # Combine DataFrames
    final_heineken_brand_count_df, final_competitor_brand_count_df = combine_detect_df(
        heineken_brand_count_df_1, competitor_brand_count_df_1, heineken_brand_count_df, heineken_brand_count_df)


    if len(all_count_df) == 0:
        all_count_df = None
    if len(final_heineken_brand_count_df) == 0:
        final_heineken_brand_count_df = None
    if len(final_competitor_brand_count_df) == 0:
        final_competitor_brand_count_df = None

    return scene_hashtags, enhanced_description, yolo_df, all_count_df, final_heineken_brand_count_df, final_competitor_brand_count_df