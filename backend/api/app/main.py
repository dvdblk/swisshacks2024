from fastapi import FastAPI, File, UploadFile
import redis
import io
import uuid
import asyncio
import json
import librosa

app = FastAPI()
redis_client = redis.Redis(host="redis", port=6379, db=0)
pubsub = redis_client.pubsub()


@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """claude generated code mostly, kinda bad can't even timeout properly if the
    ml container errors out but it's fine for now
    """
    contents = await file.read()
    # Load audio file using librosa
    audio, sr = librosa.load(io.BytesIO(contents), sr=16000)
    # create a unique job id
    job_id = str(uuid.uuid4())

    # Store the audio data in Redis
    redis_client.set(f"audio:{job_id}", audio.tobytes())
    redis_client.set(f"sr:{job_id}", str(sr))

    pubsub.subscribe(f"result:{job_id}")

    redis_client.lpush("audio_jobs", job_id)

    completed_tasks = 0

    result = {
        "rec_id": file.filename,
    }
    while (
        completed_tasks < 2
    ):  # We're waiting for results from two methods as they can be processed concurrently
        message = await asyncio.to_thread(pubsub.get_message, timeout=10)
        if message and message["type"] == "message":
            message_data = json.loads(message["data"])

            # Parse the result
            method_result = message_data["result"]
            if message_data["method"] == "left_branch":
                result["transcript"] = method_result[0]
                result["reasoning"] = method_result[1]
                result["is_factually_correct"] = method_result[2]
            else:
                result["is_fake"] = method_result[0]
                result["is_impersonator"] = method_result[1]

            completed_tasks += 1

    pubsub.unsubscribe(f"result:{job_id}")

    redis_client.delete(f"audio:{job_id}")
    redis_client.delete(f"sr:{job_id}")

    if completed_tasks == 2:
        return result
    else:
        return {"error": "Processing timed out or failed"}
