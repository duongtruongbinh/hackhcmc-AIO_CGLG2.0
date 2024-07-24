from fastapi import APIRouter
from typing import List
from pydantic import BaseModel
import httpx
from .multi_image_api import multi_image
router = APIRouter()


class ImageInput(BaseModel):
    img_base64_list: List[str]
    options: List[str]


@router.post("/multi_image/")
async def multi_image1(input: ImageInput):
    print(f"Length of img_base64_list: {len(input.img_base64_list)}")
    scene_hashtag_list, content = await multi_image(input.img_base64_list, input.options)
    print(scene_hashtag_list)
    return {"scene_hashtag_list": str(scene_hashtag_list),
            "content": str(content)}

