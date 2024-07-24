from fastapi import APIRouter
from pydantic import BaseModel
from .clip_api import classify_image_clip
router = APIRouter()


class ImageInput(BaseModel):
    base64_string: str


@router.post("/image_classifier/")
async def classifi_image(input: ImageInput):
    location, environment = classify_image_clip(input.base64_string)
    return {"location": location, "environment": environment}

ImageInput.model_rebuild = ImageInput.model_json_schema()

