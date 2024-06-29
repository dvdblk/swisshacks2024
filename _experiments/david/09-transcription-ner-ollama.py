"""
Use below to run this in the bg.

```
nohup python 09-transcription-ner-ollama.py > fact_check.log 2> fact_check_error.log &
```

"""

from dotenv import load_dotenv
import os
import json
import csv
import requests

load_dotenv()


# Function to interact with Ollama API
def generate_text(prompt, model_name="gemma2:9b-instruct-fp16"):
    url = os.getenv("OLLAMA_API_URL")
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            # fyi this param is a number of gpu "layers", not exactly the number of gpus
            "num_gpu": 26,
        },
    }
    response = requests.post(url, json=data)
    return response.json()["response"]


def check_transcript(transcript, client_features_string):
    # Define the check_transcript function
    INSTRUCTIONS = """You are an AI assistant that analyzes transcripts and compares them to a database of people. ALWAYS provide your responses in the following JSON format:
    {
    "is_factually_correct": "<yes|no>",
    "reasoning": "<Your short step-by-step reasoning here>"
    }
    Ensure your response can be parsed as valid JSON (it has to start and end with curly braces and nothing else). Do not include any text outside of this JSON structure in your response."""

    prompt = f"""{INSTRUCTIONS}
    Here is a database of people:
    {client_features_string}
    And here is a transcript:
    "{transcript}"
    Is this transcript factually correct according to the database? Analyze the data and provide your response in the requested JSON format.
    """
    response = generate_text(prompt)
    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw_output": response}


# Load client features
with open(os.getenv("DATA_PATH") + "/client_features.csv", "r") as f:
    client_features_string = f.read()

# Load transcripts
transcripts = {}
with open(os.getenv("DATA_PATH") + "/transcriptions.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        transcripts[row["Audio File"]] = row["Transcription"]

# delete the fact_checks.csv file if it already exists
if os.path.exists(os.getenv("DATA_PATH") + "/fact_checks.csv"):
    os.remove(os.getenv("DATA_PATH") + "/fact_checks.csv")


import logging

# Configure logging to output to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


# for each transcript, get the response from ollama and save it as a fact_check for the audio file
fact_checks = {}
for audio_file, transcript in transcripts.items():
    fact_checks[audio_file] = check_transcript(transcript, client_features_string)

    # Replace print with logging
    logging.info(audio_file, fact_checks[audio_file])

    # append the fact_check to a csv file after each iteration
    with open(os.getenv("DATA_PATH") + "/fact_checks.csv", "a") as file:
        writer = csv.DictWriter(
            file, fieldnames=["Audio File", "is_factually_correct", "reasoning"]
        )

        writer.writerow(
            {
                "Audio File": audio_file,
                "is_factually_correct": fact_checks[audio_file]["is_factually_correct"],
                "reasoning": fact_checks[audio_file]["reasoning"],
            }
        )
