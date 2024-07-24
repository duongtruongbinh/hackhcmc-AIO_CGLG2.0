from fastapi import APIRouter
from pydantic import BaseModel
import uvicorn
from .od_api import main_yolo
router = APIRouter()


class ImageInput(BaseModel):
    base64_string: str


@router.post("/object_detection/")
async def object_detection(input: ImageInput):
    final_df, all_count_df, heineken_brand_count_df, competitor_brand_count_df = main_yolo(
        input.base64_string)
    final_df = final_df.to_dict(orient="records")

    return {"info_od": str(final_df),
            "all_count_df": str(all_count_df),
            "heineken_brand_count_df": str(heineken_brand_count_df),
            "competitor_brand_count_df": str(competitor_brand_count_df)}

