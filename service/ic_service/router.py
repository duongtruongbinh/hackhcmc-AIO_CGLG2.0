from fastapi import APIRouter
from pydantic import BaseModel
import uvicorn
from .ic_api import ic_func
router = APIRouter()


class ImageInput(BaseModel):
    base64_string: str
    location: str
    options: list


@router.post("/image_captioning/")
async def image_captioning(input: ImageInput):
    ic = ic_func(input.base64_string, input.location, input.options)
    print(ic)
    return {"ic": str(ic)}


