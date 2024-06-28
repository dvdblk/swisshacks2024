from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings, SettingsConfigDict
import shutil
import os
from typing import Optional


class Settings(BaseSettings):
    API_KEY: str
    API_PORT: int

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

app = FastAPI()

API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: Optional[str] = Depends(api_key_header)):
    if api_key_header == settings.API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")


@app.post("/upload-audio/")
async def upload_audio(
    audio_file: UploadFile = File(...), api_key: str = Depends(get_api_key)
):
    try:
        # Create a temporary file to store the uploaded audio
        with open(f"{audio_file.filename}", "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # Here you can add your audio processing logic
        # For now, we'll just return a success message

        return JSONResponse(
            content={
                "filename": audio_file.filename,
                "message": "Audio file uploaded successfully",
            },
            status_code=200,
        )

    except Exception as e:
        return JSONResponse(
            content={"message": f"An error occurred: {str(e)}"}, status_code=500
        )

    finally:
        # Clean up the temporary file
        if os.path.exists(audio_file.filename):
            os.remove(audio_file.filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.API_PORT)
