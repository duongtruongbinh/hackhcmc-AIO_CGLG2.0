from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from clip_service.router import router as clip_router
from ic_service.router import router as ic_router
from multi_img.router import router as multi_router
from object_detection_service.router import router as ob_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.include_router(clip_router)
app.include_router(ic_router)
app.include_router(multi_router)
app.include_router(ob_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn gốc
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả các headers
)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)