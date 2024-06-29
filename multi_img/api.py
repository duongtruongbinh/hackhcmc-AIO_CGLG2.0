from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import uvicorn
from multi_image_api import multi_image, convert_to_base64
app = FastAPI()


class ImageInput(BaseModel):
    img_path: List[str]
    options: List[str]


@app.post("/multi_image/")
async def multi_image1(input: ImageInput):
    img_base64_list = convert_to_base64(input.img_path)
    scene_hashtag_list, content, _, _, _, _ = multi_image(
        img_base64_list, input.options)
    return {"scene_hashtag_list": str(scene_hashtag_list),
            "content": str(content)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
