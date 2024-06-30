from fastapi import APIRouter, File, UploadFile
import torch
import io
import uuid
import asyncio

router = APIRouter()


@router.get("/ping")
def ping():
    """Health check endpoint"""
    return {"ping": "pong"}


@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    # Read the audio file
    contents = await file.read()

    # Convert to tensor (adjust this based on your specific needs)
    audio_tensor = torch.load(io.BytesIO(contents))

    # Generate a unique job ID
    job_id = str(uuid.uuid4())

    # Store the tensor in Redis
    redis_client.set(f"audio:{job_id}", audio_tensor.numpy().tobytes())

    # Subscribe to the result channel
    pubsub.subscribe(f"result:{job_id}")

    # Notify the GPU worker
    redis_client.lpush("audio_jobs", job_id)

    # Wait for the result
    for message in pubsub.listen():
        if message["type"] == "message":
            # Unsubscribe to clean up
            pubsub.unsubscribe(f"result:{job_id}")

            # Retrieve and parse the result
            result_bytes = redis_client.get(f"result:{job_id}")
            result = torch.from_numpy(np.frombuffer(result_bytes, dtype=np.float32))

            # Clean up
            redis_client.delete(f"audio:{job_id}")
            redis_client.delete(f"result:{job_id}")

            return {"job_id": job_id, "result": result.tolist()}

    # If we get here, something went wrong
    return {"error": "Processing timed out or failed"}


# Helper function to run blocking Redis operations in a separate thread
async def redis_listener(channel):
    while True:
        message = await asyncio.to_thread(channel.get_message, timeout=1)
        if message:
            return message
