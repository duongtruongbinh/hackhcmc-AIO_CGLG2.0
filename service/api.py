from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
import uvicorn
from clip_service.router import router as clip_router
from ic_service.router import router as ic_router
from multi_img.router import router as multi_router
from object_detection_service.router import router as ob_router

app = FastAPI()
app.include_router(clip_router)
app.include_router(ic_router)
app.include_router(multi_router)
app.include_router(ob_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)