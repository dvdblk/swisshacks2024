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

# FIXME: structlog below and move the stuff above to a main func or smth
print("Models loaded")


def get_transcript(audio_tensor):
    """Invoke smol Whisper inference to get the transcript of the audio in English."""
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
