from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from ic_api import ic_func
app = FastAPI()


class ImageInput(BaseModel):
    base64_string: str
    location: str
    options: list


@app.post("/image_captioning/")
async def image_captioning(input: ImageInput):
    ic = ic_func(input.base64_string, input.location, input.options)
    print(ic)
    return {"ic": str(ic)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
