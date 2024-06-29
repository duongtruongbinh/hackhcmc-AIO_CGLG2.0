from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from clip_api import classify_image_clip
app = FastAPI()


class ImageInput(BaseModel):
    base64_string: str


@app.post("/image_classifier/")
async def classifi_image(input: ImageInput):
    location, environment = classify_image_clip(input.base64_string)
    return {"location": location, "environment": environment}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
