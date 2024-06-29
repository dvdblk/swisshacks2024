from fastapi import FastAPI, File, UploadFile
import pandas as pd
import os

app = FastAPI()

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    contents = await file.read()
    file_path = f"data/{file.filename}"
    with open(file_path, 'wb') as f:
        f.write(contents)
    return {"filename": file.filename, "path": file_path}

@app.get("/")
def read_root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    if not os.path.exists('data'):
        os.makedirs('data')
    uvicorn.run(app, host="0.0.0.0", port=8000)
