from typing import Tuple

import redis
import torch
import numpy as np
import time
import json
import asyncio


redis_client = redis.Redis(host="redis", port=6379, db=0)

# load the whisper model here
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-tiny"

whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
whisper_model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

print("Models loaded")


def _get_transcript(audio_tensor):

    pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=256,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    transcript = pipe(audio_tensor.numpy(), generate_kwargs={"language": "english"})
    return transcript["text"].lstrip()


def process_audio_is_factually_correct(audio_tensor) -> Tuple[str, str, False]:
    """
    Left branch of the pipeline (produces `is_factually_correct` and some othe metadata e.g. reasoning)

    Note:
        1) run whisper to get transcript
        2) extract entities (facts) from the transcript via LLM call
        3) fuzzy match the name of the entity with the known speakers in the db
        4) call LLM again with full row of the database to compare with the transcript
        5) return True if the entity is factually correct, False otherwise, along with reasoning

    Returns:
        Tuple[str, str, bool]: (transcript, reasoning, is_factually_correct)
    """
    # FIXME: remove uggo try-except any to prevent api from waiting infinitely
    try:
        transcript = _get_transcript(audio_tensor)

        # extract entities from the transcript via LLM call

        return [transcript, "this is the reasoning", False]  # Return two booleans
    except Exception as e:
        # FIXME: structlog
        print(e)
        return [None, None, None]


def process_audio_is_fake_is_impersonator(audio_tensor) -> Tuple[bool, bool]:
    """
    Right branch of the pipeline (produces `is_fake` and `is_impersonator`)

    Note:
        1)
    """
    time.sleep(2)  # Simulate GPU processing
    return [True, True]  # Return one boolean


async def process_job(job_id):
    audio_bytes = redis_client.get(f"audio:{job_id}")
    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
    audio_tensor = torch.from_numpy(audio_array.copy())

    # Process the audio with both methods concurrently
    result1_task = asyncio.create_task(
        asyncio.to_thread(process_audio_is_factually_correct, audio_tensor)
    )
    result2_task = asyncio.create_task(
        asyncio.to_thread(process_audio_is_fake_is_impersonator, audio_tensor)
    )

    result1 = await result1_task
    result2 = await result2_task

    # Publish results
    redis_client.publish(
        f"result:{job_id}", json.dumps({"method": "left_branch", "result": result1})
    )
    redis_client.publish(
        f"result:{job_id}", json.dumps({"method": "method2", "result": result2})
    )

    # Clean up
    redis_client.delete(f"audio:{job_id}")


async def main():
    while True:
        _, job_id = await asyncio.to_thread(redis_client.brpop, "audio_jobs")
        job_id = job_id.decode("utf-8")
        await process_job(job_id)


if __name__ == "__main__":
    asyncio.run(main())
