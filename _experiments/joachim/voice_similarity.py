"""
copied from notebook to this python file

run in directory:
!apt-get -y -q install libsox-dev
!pip install git+https://github.com/openai/whisper.git
!pip install asteroid-filterbanks
!pip install librosa
!pip install git+https://github.com/pyannote/pyannote-audio

"""

import datetime
import subprocess

import whisper
import torch
import pyannote.audio
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=device)

from pyannote.audio import Audio
from pyannote.core import Segment
import librosa
import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import sklearn
from tqdm import tqdm
import csv

num_speakers = 20

language = 'any' #@param ['any', 'English']

model_size = 'large' #@param ['tiny', 'base', 'small', 'medium', 'large']

"""
!mkdir similarity_metrics_tiny
!mkdir similarity_metrics_large
"""

real_labels = ["PASOTPBNLM",
               "H01FH3KEY8",
               "BJ53WB0WQB",
               "VWEQSW9GQY",
               "R18W797V9Q",
               "6W361V5VV9",
               "ZBKI0P43EK",
               "ROHDD0Z6CG",
               "2IM42LTT5R",
               "Y6K2JU2H4B",
               "CBDX295MEZ",
               "ZCB53KC2PC",
               "3162VQ31V7",
               "YDDKTODLGX",
               "XKG8C7QFXT",
               "1Z2W0U9OU8",
               "ZGZHPG1TS8",
               "244F8XZK0E",
               "7FPDGERPRV",
               "IIBWPCAJFZ"]
fake_labels = ["NSIOUFFN5C",
               "MIRV2AHSDH",
               "GE90UVYAIC",
               "GINYUH6NU7",
               "ASLJ66JRJL",
               "2TT75RT0RO",
               "FG1GU97VU5",
               "FI3U0S0S6X",
               "RLL2WGXJRT",
               "IU50O8RY55",
               "0D8CAOL7XN",
               "541T0I3AUW",
               "B5PN7WKKMI",
               "F68PGID9TU",
               "TC1N3OMAN3",
               "H6XNGJ7SCM",
               "9PS130EZ8T",
               "X8L6WJ0NDN",
               "7A8PVRXFLV",
               "12MINIG2V7",
               ]

segments = []
#parent_durations = []
#parent_paths = []
model = whisper.load_model(model_size)
model.to(device)
model.eval()
for real_recording_id in real_labels:

    path = f"/kaggle/input/swisshacks/{real_recording_id}.wav"
    result = model.transcribe(path)
    sample_segments = result["segments"]

    y, sr = librosa.load(path,sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    for i in range(len(sample_segments)):
        sample_segments[i]["parent_duration"]=duration
        sample_segments[i]["parent_path"] = path
        sample_segments[i]["recording_id"] = real_recording_id

    segments.extend(sample_segments)

audio = Audio()

def segment_embedding(segment):
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(segment["parent_duration"], segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(segment["parent_path"], clip)
    return embedding_model(waveform[None])

embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
    embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

recording_to_segment_mapping = dict()
segment_to_user_id_mapping=[]
# could use defaultdict
for k in real_labels:
    recording_to_segment_mapping[k] = []
for s_idx, s in enumerate(segments):
    recording_to_segment_mapping[s["recording_id"]].append(s_idx)
    segment_to_user_id_mapping.append(real_labels.index(s["recording_id"]))

# testing
train_indices = []
test_indices = []
for k in real_labels:
    train_indices.extend(recording_to_segment_mapping[k][:-1])
    test_indices.append(recording_to_segment_mapping[k][-1])

non_single_self_similarities = []
minimum_distance_other_user = []
for k in real_labels:
    #print(len(recording_to_segment_mapping[k])
    test_index = recording_to_segment_mapping[k][-1]
    similarities = sklearn.metrics.pairwise.pairwise_distances(embeddings, embeddings[[test_index]], metric='cosine', n_jobs=1)
    #print("*"*3,k,"*"*3)
    start_idx = 0
    min_neighb = 2.
    for l in real_labels:
        num_segments = len(recording_to_segment_mapping[l])
        sub = 0
        if num_segments>1:
            sub=1

        mean_similarity = similarities[start_idx:start_idx+(num_segments-sub)].mean()

        if k == l:
            #print("SELF Distance: ", mean_similarity)
            if num_segments >1:
                non_single_self_similarities.append(mean_similarity)
        else:
            #print("similarity: ",mean_similarity )
            if mean_similarity < min_neighb:
                min_neighb=mean_similarity
        start_idx += num_segments
    minimum_distance_other_user.append(min_neighb)

#non_single_self_similarities
import os
import fnmatch
unlabled_sample_paths = []
pattern = '*.wav'
# Iterate over files in the directory
for file in os.listdir("/kaggle/input/swisshacks/"):
    if fnmatch.fnmatch(file, pattern):
        # Do something with the file
        #print(file)  # Example: Print the filename
        label = 2
        if file[:-4] in real_labels:
            label = 0
            #print("real:", file, "skipping")
        elif file[:-4] in fake_labels:
            label = 1
            #print("fake:", file, "skipping")
        else:
            unlabled_sample_paths.append(file)
print(len(unlabled_sample_paths))


## Save Real Samples Embeddings
np.savez(f"similarity_metrics_{model_size}/gt_embeddings.npz", gt_embeddings=embeddings, segment_user_id=segment_to_user_id_mapping)

# Calculate User Similarity for Unlabeled Data

# Main processing step
for f_idx, unlabled_recording_file in tqdm(enumerate(unlabled_sample_paths),total=len(unlabled_sample_paths)):

    path = f"/kaggle/input/swisshacks/{unlabled_recording_file}"
    result = model.transcribe(path)
    sample_segments = result["segments"]

    y, sr = librosa.load(path,sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    for i in range(len(sample_segments)):
        sample_segments[i]["parent_duration"]=duration
        sample_segments[i]["parent_path"] = path
        sample_segments[i]["recording_id"] = unlabled_recording_file[:-4]
    sample_embeddings = np.zeros(shape=(len(sample_segments), 192))
    for i, segment in enumerate(sample_segments):
        sample_embeddings[i] = segment_embedding(segment)

    sample_embeddings = np.nan_to_num(sample_embeddings)
    sample_similarities = sklearn.metrics.pairwise.pairwise_distances(embeddings, sample_embeddings, metric='cosine', n_jobs=1)
    #print(sample_similarities)
    np.savez(f"similarity_metrics_{model_size}/{unlabled_recording_file[:-4]}.npz", sim=sample_similarities, segment_user_id=segment_to_user_id_mapping)

# Export to CSV

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["ID"]+[f"USER_{i}_{j}" for i,j in enumerate(real_labels)])
    # Iterate over files in the directory
    for file in os.listdir(f"/kaggle/working/similarity_metrics_{model_size}"):
        if fnmatch.fnmatch(file, pattern):

            # Do something with the file
            #print(file)  # Example: Print the filename
            scores = []
            with np.load(os.path.join(f"/kaggle/working/similarity_metrics_{model_size}",file)) as data:
                if "gt_embeddings" in data:
                    print("gt")
                    continue
                sim = data['sim']
                for user_idx in range(20):
                    segement_indices = recording_to_segment_mapping[real_labels[user_idx]]
                    #print(min(segement_indices),max(segement_indices))
                    scores.append(sim[min(segement_indices):max(segement_indices)+1].mean())
            #print(scores)
            writer.writerow([file[:-4]]+scores)

"""
0.37 seems like a reasonable threshold for maximum self-distance to turn into alert zone
"""
