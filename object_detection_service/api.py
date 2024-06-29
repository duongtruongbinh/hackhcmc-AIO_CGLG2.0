from utils_func import process_image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from typing import Optional
import uvicorn
from od_api import main_yolo
app = FastAPI()


class ImageInput(BaseModel):
    base64_string: str


@app.post("/object_detection/")
async def object_detection(input: ImageInput):
    final_df, all_count_df, heineken_brand_count_df, competitor_brand_count_df = main_yolo(
        input.base64_string)
    final_df = final_df.to_dict(orient="records")

    return {"info_od": str(final_df),
            "all_count_df": str(all_count_df),
            "heineken_brand_count_df": str(heineken_brand_count_df),
            "competitor_brand_count_df": str(competitor_brand_count_df)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
