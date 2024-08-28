'''
Script to split audio metadata from csv file into n chunks. 
Set n to the desired number of chunks based on the number of available resources. If only one chunk is desired, set n=1.
Run as:
    python transcription/split_audio.py --n 8
'''

import os.path as osp
import pandas as pd
from copy import deepcopy
import argparse

import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from utils.setup import *

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1, help="Number of chunks to split into")
parser.add_argument("--csv_file", type=str, default="bot.csv", help="CSV file containing metadata")
args = parser.parse_args()

# load metadata
model_name = "large-v2"
metadata = pd.read_csv(osp.join(metadata_dir, model_name, "episodes", args.csv_file))
print(f"Number of total videos: {len(metadata)}")

# transcribed audio
json_dir = osp.join(transcript_dir, "json", model_name, "files")
if not osp.exists(json_dir):
    print(f"Directory {json_dir} does not exist. Creating...")
    os.makedirs(json_dir, exist_ok=True)
transcribed_epids = [fname.split('.')[0] for fname in os.listdir(json_dir)]
print(f"Number of transcribed videos: {len(transcribed_epids)}")

# metadata remaining to transcribe
metadata = metadata[~metadata['ep_id'].isin(transcribed_epids)]
print(f"Number of videos to transcribe: {len(metadata)}")

if len(metadata) == 0:
    print("No videos to transcribe.")
    exit()

# split into n chunks
savedir = osp.join(metadata_dir, model_name, "episodes", "splits")
os.system(f'rm -rf {savedir}')
os.makedirs(savedir, exist_ok=True)
chunk_size = len(metadata) // args.n
remainder = len(metadata) % args.n

for i in range(args.n):
    if i == 0:
        start = i * chunk_size
    else:
        start = deepcopy(end)
    end = start + chunk_size
    if remainder > 0:
        end += 1
        remainder -= 1
    print(f"Chunk {i}: {start} to {end}")
    metadata_chunk = metadata.iloc[start:end]
    metadata_chunk.to_csv(osp.join(savedir, f"episodes_{i}.csv"), index=False)
    print(f"Saved episodes_{i}.csv.  Length: {len(metadata_chunk)}")
