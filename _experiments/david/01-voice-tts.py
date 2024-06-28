import torch
from TTS.api import TTS
from pydub import AudioSegment
import numpy as np


def load_audio(file_path):
    """Load audio file and convert to numpy array."""
    audio = AudioSegment.from_file(file_path)
    samples = np.array(audio.get_array_of_samples())
    return samples.astype(np.float32) / 32768.0  # Normalize to [-1, 1]


def clone_voice(model, reference_audio, text):
    """Clone voice and generate speech."""
    return model.tts(text=text, speaker_wav=reference_audio)


def save_audio(audio, file_path):
    """Save audio array to file."""
    audio = (audio * 32768).astype(np.int16)
    audio_segment = AudioSegment(
        audio.tobytes(), frame_rate=22050, sample_width=2, channels=1
    )
    audio_segment.export(file_path, format="wav")


def voice_cloning_pipeline(reference_audio_files, texts, output_dir):
    # Initialize YourTTS model
    model = TTS("tts_models/multilingual/multi-dataset/your_tts").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    for i, (audio_file, text) in enumerate(zip(reference_audio_files, texts)):
        # Load reference audio
        reference_audio = load_audio(audio_file)

        # Clone voice and generate speech
        cloned_speech = clone_voice(model, reference_audio, text)

        # Save the generated speech
        output_file = f"{output_dir}/cloned_speech_{i}.wav"
        save_audio(cloned_speech, output_file)
        print(f"Generated cloned speech saved to {output_file}")


# Example usage
reference_audio_files = ["speaker1.wav", "speaker2.wav", "speaker3.wav"]
texts = [
    "This is a test of voice cloning technology.",
    "Voice cloning has many potential applications.",
    "We should use this technology responsibly.",
]
output_dir = "cloned_voices"

voice_cloning_pipeline(reference_audio_files, texts, output_dir)
