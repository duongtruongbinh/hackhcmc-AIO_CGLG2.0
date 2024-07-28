from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from predict_system import process
from extract_number import output_ocr
app = FastAPI()


class ImageInput(BaseModel):
    base64_string: str

@app.post("/process_image/")
async def process_image(input: ImageInput):
    ocr_re = process(input.base64_string)
    ocr_re1 = ocr_re[0]["res"]
    if len(ocr_re1) == 0:
        ocr_re1 = [{'text': "CGLG2.0", "confidence": 0.99, "text_region": [[788.0, 70.0], [1477.0, 68.0], [1477.0, 126.0], [788.0, 128.0]]}]
    print(ocr_re1)
    # df = pd.DataFrame(ocr_re1)
    # print(df)
    # print(type(df))
    return_count = output_ocr(ocr_re)
    # Return a success response
    return {"ocr_result": str(ocr_re1), "count": return_count}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
