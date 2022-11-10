from typing import Union
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    img = await file.read()
    return StreamingResponse(io.BytesIO(img), media_type="image/png")