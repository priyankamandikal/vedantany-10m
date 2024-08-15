
"""
Dataset class for creating and loading chunks.
chunk loader used by embed_chunks.py
Run as:
    python create_chunks.py --csv_file bot.csv --chunk_size 1500 --chunk_overlap 0
"""

import pandas as pd
import json
import shutil
from tqdm import tqdm
from torch.utils.data import Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.setup import *
from utils.chunk_matcher import ChunkMatcher


class ChunkDataset(Dataset):

    def __init__(self, whisper_model="large-v2", csv_file=None, create_chunks=False, chunk_size=1500, chunk_overlap=0):
        self.whisper_model = whisper_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        csv_file = osp.join(metadata_dir, self.whisper_model, 'episodes', csv_file)

        # Read the metadata
        self.stor_metadata = pd.read_csv(csv_file, index_col=None)
        print("Number of episodes:", len(self.stor_metadata))
        
        # Load chunks
        if create_chunks:
            self.splits, self.metadatas = self.create_chunks()
        else:
            self.splits, self.metadatas = self.load_chunks()

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, idx):
        return self.splits[idx], self.metadatas[idx]
    
    def create_chunks(self):
        print("Creating chunks...")
        chunk_dir = osp.join(CHUNK_DIR, self.whisper_model)
        if osp.exists(chunk_dir):
            shutil.rmtree(chunk_dir)
        os.makedirs(chunk_dir)
        
        splits_all = [ ]
        metadatas_all = [ ]
        for ix in tqdm(range(len(self.stor_metadata))):
        # for ix in self.stor_metadata.index:

            # # terminate after 2 episodes
            # if ix == 2:
            #     break

            # Get data
            title = self.stor_metadata.loc[ix,'title']
            ep_link = self.stor_metadata.loc[ix,'link']
            ep_id = ep_link.split("=")[1]
            playlist_id = self.stor_metadata.loc[ix,'playlist_id']

            # Get transcript
            try:
                tmp_out_fpath = osp.join(transcript_dir, "json", "tmp.txt")
                with open(osp.join(transcript_dir, "json", self.whisper_model, "playlists", playlist_id, "%s.json"%ep_id), "r") as f_in: # large-v2
                    segments = json.load(f_in)['segments']
                    with open(tmp_out_fpath, "w") as f_out:
                        for seg in segments:
                            ts = int(seg['start'])
                            f_out.write(self.stor_metadata.loc[ix,'link'] + "&t=%s"%ts + "\t" + str(ts) + "\t" + seg['text'] + "\n")
            except Exception as e:
                print(title, ep_link)
                print(e)
                continue

            # Read transcript 
            transcript = pd.read_csv(tmp_out_fpath,sep='\t',header=None)
            transcript.columns = ['links','time','chunks']
            
            # Clean text chunks 
            transcript['clean_chunks'] = transcript['chunks'].astype(str).apply(lambda x: x.strip())
            links = list(transcript['links'])
            texts = transcript['clean_chunks'].str.cat(sep=' ')

            # Splits 
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, 
                                                           chunk_overlap=self.chunk_overlap) 
            splits = text_splitter.split_text(texts)
            # if the last split is too short (< 300 characters), then merge it with the previous split
            if len(splits[-1]) < 300:
                splits[-2] = splits[-2] + splits[-1]
                splits = splits[:-1]
            # print(len(splits)) 

            # Metadata 
            chunk_matcher = ChunkMatcher(splits, transcript['clean_chunks'], links, searchlen=5, verbose=False)
            watch_links, embed_links = chunk_matcher.get_matching_links()
            
            # Here we can add "link", "title", etc that can be fetched in the app 
            metadatas=[{"source" : title + " " + link,
                        "playlist_id": playlist_id, 
                        "ep_id": ep_id,
                        "link": link,
                        "embed_link": embed_link,
                        "title": title
                        } for link, embed_link in zip(watch_links, embed_links)]

            # save all splits and metadatas in a json
            with open(osp.join(chunk_dir, f'{ep_id}.json'), 'w') as f:
                json.dump([{'text': split, 'metadata': metadata} for split, metadata in zip(splits, metadatas)], f)

            # Append to output 
            splits_all.append(splits)
            metadatas_all.append(metadatas)
        print("... Done")

        # Flatten the list of lists
        splits_all = [item for sublist in splits_all for item in sublist]
        metadatas_all = [item for sublist in metadatas_all for item in sublist]

        return splits_all, metadatas_all
    
    def load_chunks(self):
        print("Loading chunks...")
        chunk_dir = osp.join(CHUNK_DIR, self.whisper_model)
        splits_all = [ ]
        metadatas_all = [ ]
        print("Number of episodes:", len(self.stor_metadata))
        for ix in self.stor_metadata.index:
            ep_link = self.stor_metadata.loc[ix,'link']
            ep_id = ep_link.split("=")[1]
            with open(osp.join(chunk_dir, f'{ep_id}.json'), 'r') as f:
                data = json.load(f)
            splits = [item['text'] for item in data]
            metadatas = [item['metadata'] for item in data]
            splits_all.extend(splits)
            metadatas_all.extend(metadatas)
        print("... Done")
        print("Number of chunks:", len(splits_all))
        return splits_all, metadatas_all
    

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    # Chunking params
    parser.add_argument("--whisper_model", type=str, default="large-v2", help="Whisper model to use. Options: large-v2")
    parser.add_argument("--csv_file", type=str, default="bot.csv", help="CSV file containing metadata")
    parser.add_argument("--chunk_size", type=int, default=1500, help="Number of characters in chunk")
    parser.add_argument("--chunk_overlap", type=int, default=0, help="Number of characters to overlap between chunks")
    args = parser.parse_args()
    
    dataset = ChunkDataset(whisper_model=args.whisper_model,
                           csv_file=args.csv_file, 
                           create_chunks=True, 
                           chunk_size=args.chunk_size, 
                           chunk_overlap=args.chunk_overlap)
    