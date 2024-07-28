from fastapi import APIRouter
from pydantic import BaseModel
import uvicorn
from .ic_api import ic_func_openAI
router = APIRouter()


class ImageInput(BaseModel):
    base64_string: str
    location: str
    options: list


@router.post("/image_captioning/")
async def image_captioning(input: ImageInput):
    scene_hashtags, enhanced_description, yolo_df, all_count_df, heineken_brand_count_df, competitor_brand_count_df = await ic_func_openAI(input.base64_string, input.location, input.options)
    return {"scene_hashtags": scene_hashtags,
            "enhanced_description": enhanced_description,
            "yolo_df": str(yolo_df),
            "all_count_df": str(all_count_df),
            "heineken_brand_count_df": str(heineken_brand_count_df),
            "competitor_brand_count_df": str(competitor_brand_count_df)}
    


