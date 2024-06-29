from dotenv import load_dotenv
import os
import json
import csv
import requests
import pandas as pd
from rapidfuzz import fuzz

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


# Define the check_transcript function
def check_transcript(transcript):
    INSTRUCTIONS = """You are an AI assistant that extracts important facts from a transcripts of a phone call to a bank. ALWAYS provide your responses in the following JSON format which contains the information you need to extract:
    {
    "name": <the name of the person speaking, always comes first in the transcript>,
    "birthday": <(optional) the birthdate of the person>,
    "marital_status": <(optional) the marital status of the person>,
    "account_nr": <(optional) the account number of the person>,
    "tax_residency": <(optional) the tax residency of the person>,
    "net_worth_in_millions": <(optional) the net worth of the person>,
    "profession": <(optional) the profession of the person>,
    "social_security_number": <(optional) the social security number of the person>,
    "relationship_manager": <(optional) the relationship manager of the person>,
    "highest_previous_education": <(optional) the highest previous education of the person>,
    }
    Ensure your response can be parsed as valid JSON (it has to start and end with curly braces and nothing else). Do not include any text outside of this JSON structure in your response.
    """
    prompt = f"""{INSTRUCTIONS}

    And here is the transcript:
    "{transcript}"
    Extract the key facts from the transcript, account for transcription errors and try to correct them. Provide your response in the requested JSON format.
    """
    response = generate_text(prompt)
    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw_output": response}


# Load client features
with open(os.getenv("DATA_PATH") + "/client_features.csv", "r") as f:
    client_features = pd.read_csv(f)

    # lowercase every string in the database
    # client_features = client_features.applymap(lambda s: s.lower() if type(s) == str else s)

    # remove all symbols from social security numbers except numbers
    client_features["social_security_number"] = client_features[
        "social_security_number"
    ].str.replace(r"\D", "", regex=True)

    # convert to string
    client_features_string = client_features.astype(str)


# Load transcripts
transcripts = {}
with open(os.getenv("DATA_PATH") + "/transcriptions.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        transcripts[row["Audio File"]] = row["Transcription"]


df = pd.read_csv(os.getenv("DATA_PATH") + "/client_features.csv")
# remove all symbols from social security numbers except numbers
df["social_security_number"] = df["social_security_number"].str.replace(
    r"\D", "", regex=True
)

query = {
    "name": "Mia Anderson",
    "highest_previous_education": "Bachelor of Science in Computer Science",
    "relationship_manager": "Ella Morrison",
}


def best_match(df, query):
    # get the row with the best matching name

    name_scores = df.apply(lambda row: fuzz.ratio(row["name"], query["name"]), axis=1)
    best_name_match_row = df.iloc[name_scores.idxmax()]

    return best_name_match_row


def check_if_row_matches(transcript, matched_person_string):
    INSTRUCTIONS = """You are an AI assistant that needs to verify whether a transcript of a phone call matches the json record. ALWAYS provide your responses in the following JSON format:
    {
        "is_matching_person": "<yes|no>",
        "reasoning": "<short reasoning>"
    }
    """
    prompt = f"""{INSTRUCTIONS}

    Here is the transcript of the phone call:
    "{transcript}"
    And here is the factual real information in a json format:
    {matched_person_string}
    Is the transcript factually correct according to the json i.e. is it the same person? While checking for fact correctness you must account for some transcription errors or typos e.g. homophones (e.g. i vs y, kh vs k), additional whitespace especially with names and surnames because they might not match exactly. Transcription errors also happen with social security numbers: sometimes there are dashes, hyphens or dots - ignore them when comparing to the number.
    Marital status has to match almost exactly, not a different word. Tax residency has to be the same area.
    Analyze the data and provide your response in the requested JSON format without backticks.
    """

    response = generate_text(prompt)
    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw_output": response}


import logging

# Configure logging to output to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


fact_checks = {}
for audio_file, transcript in transcripts.items():
    extracted_facts = check_transcript(transcript)
    best_matching_row = best_match(df, extracted_facts)
    factually_correct = check_if_row_matches(transcript, best_matching_row)
    fact_checks[audio_file] = factually_correct
    logging.info(f"Finished {audio_file}")

    # append the fact_check to a csv file after each iteration
    with open(os.getenv("DATA_PATH") + "/fact_checks_v2.csv", "a") as file:
        writer = csv.DictWriter(
            file, fieldnames=["Audio File", "is_factually_correct", "reasoning"]
        )

        writer.writerow(
            {
                "Audio File": audio_file,
                "is_factually_correct": fact_checks[audio_file]["is_matching_person"],
                "reasoning": fact_checks[audio_file]["reasoning"],
            }
        )
