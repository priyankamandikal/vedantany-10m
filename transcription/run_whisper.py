'''
Script to run whisper on saved youtube audio files from all playlists.
Run as:
    python transcription/run_whisper.py --model_name large-v2 --gpu 0 --chunk 0
'''

import pytube as pt
import whisper
import json
import os
import os.path as osp
from time import time
import pandas as pd
import logging
import argparse

import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from utils.setup import *

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="large-v2", help="Name of whisper model to use.")
parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
parser.add_argument("--chunk", type=int, default=0, help="Chunk of episodes to run. Check metadatadir for chunks")
args = parser.parse_args()

# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

model_name = args.model_name

# Function to print and log messages
def printlog(message):
    logging.info(message)
    print(message)

# set up logging
log_dir = osp.join(log_dir, "whisper")
os.makedirs(log_dir, exist_ok=True)
log_file = osp.join(log_dir, f"{model_name}_{args.chunk}.log")
logging.basicConfig(filename=log_file, level=logging.INFO)

# setup dirs
save_dir = osp.join(transcript_dir, "json", model_name)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(osp.join(save_dir, "files"), exist_ok=True)
os.makedirs(osp.join(save_dir, "playlists"), exist_ok=True)

# load large whisper model
print(f"Loading {model_name} model...")
model = whisper.load_model(name=model_name, download_root=model_dir)
print("... Loaded model")

# load metadata
metadata = pd.read_csv(osp.join(metadata_dir, model_name, f"episodes/splits/episodes_{args.chunk}.csv"))

# set prompt
prompt = "In this lecture, we will discuss Vedanta philosophy. Let us begin."

if len(metadata) == 0:
    printlog("No videos to transcribe.")
    exit()

# obtain transcripts
for ix in metadata.index:
    try:
        printlog('\n' + metadata.loc[ix, "title"])
        ep_link = metadata.loc[ix, "link"]
        ep_id = ep_link.split("=")[1]
        playlist_id = metadata.loc[ix, "playlist_id"]
        os.makedirs(osp.join(save_dir, "playlists", playlist_id), exist_ok=True)
        audio_path = osp.join(audio_dir, "files", ep_id + ".mp3")
        printlog(f"Transcribing {playlist_id}/{ep_id}...")
        time_start = time()
        result = model.transcribe(audio_path, 
                                  word_timestamps=True,
                                  initial_prompt=prompt,
                                  language="en",)
        printlog(f"Transcribed in {(time() - time_start) / 60} mins\n")
        with open(osp.join(save_dir, "files", ep_id + ".json"), "w") as f:
            json.dump(result, f)
        os.symlink(f"{save_dir}/files/{ep_id}.json", f"{save_dir}/playlists/{playlist_id}/{ep_id}.json")
    except Exception as e:
        printlog("Error: " + str(e))
