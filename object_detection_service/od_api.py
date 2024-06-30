from utils_func import process_image
import json
import requests
import pandas as pd
import ast
from dotenv import load_dotenv
import os
load_dotenv()


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


def main_yolo(image_path):
    myobj = {
        "base64_string": image_path
    }

    url = os.environ.get("OCR_URL")
    x = requests.post(url, json=myobj)
    y = json.loads(x.text)
    data_list = ast.literal_eval(y["ocr_result"])
    df = pd.DataFrame(data_list)
    df = convert_OCR_result(df)
    df = df[df['confidence'] > 0.7]
    df.rename(columns={"text": "name"}, inplace=True)
    final_df, all_count_df, heineken_brand_count_df, competitor_brand_count_df = process_image(
        image_path, df)
    if all_count_df is not None:
        all_count_df = all_count_df.to_dict(orient="records")
    if heineken_brand_count_df is not None:
        heineken_brand_count_df = heineken_brand_count_df.to_dict(
            orient="records")
    if competitor_brand_count_df is not None:
        competitor_brand_count_df = competitor_brand_count_df.to_dict(
            orient="records")

    return final_df, all_count_df, heineken_brand_count_df, competitor_brand_count_df
