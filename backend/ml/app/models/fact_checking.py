from typing import Tuple
import json
import os

import requests
import pandas as pd
from rapidfuzz import fuzz


# FIXME: this should be in postgres
# load dataframes
client_features_df = pd.read_csv("/app/data/client_features.csv")
# remove all symbols from social security numbers except numbers
client_features_df["social_security_number"] = client_features_df[
    "social_security_number"
].str.replace(r"\D", "", regex=True)


# Function to interact with Ollama API
def generate_text(prompt, model_name="phi3:instruct"):
    # FIXME: pydantic settings
    url = os.getenv("OLLAMA_API_URL")
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            # Note:
            # this param is the number of gpu "layers", not exactly the number of gpus
            # needed for my 1080 + 1070 setup because im gpu poor
            "num_gpu": 26,
        },
    }
    response = requests.post(url, json=data)
    return response.json()["response"]


def _extract_facts_from_transcript(transcript):
    """
    Extracts all available facts from a transcript.

    Note: FIXME pydantic
    """
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


def _best_match(query):
    # get the row with the best matching name
    name_scores = client_features_df.apply(
        lambda row: fuzz.ratio(row["name"], query["name"]), axis=1
    )
    best_name_match_row = client_features_df.iloc[name_scores.idxmax()]

    return best_name_match_row


def fact_check_flow(transcript: str) -> Tuple[bool, str]:
    """Transcript to `is_factually_correct` and `reasoning` (why it is factually correct or not)"""
    extracted_facts = _extract_facts_from_transcript(transcript)
    try:
        best_matching_row = _best_match(extracted_facts)
        factually_correct_data = check_if_row_matches(transcript, best_matching_row)
        return (
            factually_correct_data["is_matching_person"],
            factually_correct_data["reasoning"],
        )
    except KeyError:
        return (
            "no",
            "error: failed to process the transcript for fact checking",
        )
