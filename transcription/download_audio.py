'''
Script to download audio and images from youtube playlists.
If --download-from-csv is set, it will download audio for list of videos used in the paper. If not set, it will download audio for all videos from VSNY playlists up to the current date.
Run as:
    python transcription/download_audio.py --download-from-csv
    python transcription/download_audio.py --download-from-csv --skip-saved
'''

import pandas as pd
import requests
import yt_dlp
from youtubesearchpython import *
import os
import os.path as osp
import argparse

import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from utils.setup import *

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="large-v2", help="Name of whisper model to use.")
parser.add_argument("--download-from-csv", action="store_true", help="Download audio for list of videos used in the paper")
parser.add_argument("--csv-file", type=str, default="bot.csv", help="Optional: CSV file containing list of videos to download. Set to bot.csv to use videos from the paper.")
parser.add_argument("--skip-saved", action="store_true", help="Skip downloading saved audio")
args = parser.parse_args()

playlist_ids = playlist_id_to_title.keys()
episodes_dir = osp.join(metadata_dir, args.model_name, 'episodes')
os.makedirs(episodes_dir, exist_ok=True)

def add_metadata(metadata, idx, video):
    metadata.loc[idx, 'link'] = video['ep_link']
    metadata.loc[idx, 'playlist_id'] = video['playlist_id']
    metadata.loc[idx, 'ep_id'] = video['ep_link'].split('=')[1]
    metadata.loc[idx, 'title'] = video['title']

def refresh_metadata(csv, metadata):
    if osp.exists(csv):
        print('Found old metadata. Appending new metadata.')
        old_metadata = pd.read_csv(csv)
        old_metadata.drop(old_metadata.columns[0], axis=1, inplace=True)  # drop unnamed column
        all_metadata = pd.concat([old_metadata, metadata])
        all_metadata.drop_duplicates(subset=['link'], inplace=True)
        all_metadata.reset_index(drop=True, inplace=True)
    else:
        print('No old metadata found. Saving new metadata.')
        all_metadata = metadata
    all_metadata.to_csv(csv)
    print(f'Saved metadata to {csv}')
    print('Metadata shape:', all_metadata.shape)

def download_audio(video, audio_path):
    print(video['title'])
    ep_link = video['ep_link']
    ydl_opts = {
    'format': 'mp3/bestaudio/best',
    'outtmpl': audio_path,
    'noplaylist': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(ep_link)

def download_img(video, img_path):
    img_url = video['thumbnails'][3]['url']
    with open(img_path, 'wb') as f:
        response = requests.get(img_url)
        f.write(response.content)


if __name__=='__main__':

    os.makedirs(osp.join(audio_dir, 'files'), exist_ok=True)
    os.makedirs(osp.join(audio_dir, 'playlists'), exist_ok=True)

    # delete files that end with .part in audio_dir, "files"
    for fname in os.listdir(osp.join(audio_dir, 'files')):
        if fname.endswith('.part'):
            os.remove(osp.join(audio_dir, 'files', fname))

    # get downloaded urls for all playlists
    downloaded_audio_urls = [fname.split('.')[0] for fname in os.listdir(osp.join(audio_dir, 'files')) if fname.endswith('.mp3')]
    print(f'Number of downloaded urls: {len(downloaded_audio_urls)}')
    stor_all_metadata = pd.DataFrame()
    global_idx = 0
    all_csv = osp.join(episodes_dir, 'all.csv')
    if args.download_from_csv:
        print('Downloading audio for list of videos used in the paper...')
        paper_csv = osp.join(episodes_dir, args.csv_file)
        if osp.exists(paper_csv):
            paper_metadata = pd.read_csv(paper_csv)
    video_urls = []
    if osp.exists(all_csv):
        all_metadata = pd.read_csv(all_csv)
        for idx, row in all_metadata.iterrows():
            video_urls.append(row['ep_id'])

    # Get video urls
    for playlist_id in playlist_ids:
        os.makedirs(osp.join(audio_dir, "playlists", playlist_id), exist_ok=True)
        playlist = Playlist(f'https://www.youtube.com/playlist?list={playlist_id}')
        while playlist.hasMoreVideos:
            playlist.getNextVideos()
        print(f"\n{playlist.info['info']['title']}")
        print(f'{playlist_id}: {len(playlist.videos)} videos')
        stor_metadata = pd.DataFrame()
        for idx, video in enumerate(playlist.videos):
            url = video["link"].split('&list')[0]
            ep_id = url.split('=')[1]
            if args.download_from_csv and ep_id not in paper_metadata['ep_id'].values:
                continue
            video['ep_link'] = url
            video['playlist_id'] = playlist_id
            audio_path = osp.join(audio_dir, "files", f'{ep_id}.mp3')
            audio_playlist_path = osp.join(audio_dir, "playlists", playlist_id, f'{ep_id}.mp3')
            if ep_id not in downloaded_audio_urls:
                try:
                    download_audio(video, audio_path)
                except Exception as e:
                    print(f'Failed on {playlist_id}/{ep_id}: {video["title"]}')
                    print(f'Error: {e}')
            if osp.exists(audio_path):
                if not osp.exists(audio_playlist_path):
                    os.system(f'ln -s {audio_path} {audio_playlist_path}')
                add_metadata(stor_metadata, idx, video)
                if ep_id not in video_urls:
                    add_metadata(stor_all_metadata, global_idx, video)
                    global_idx += 1
                    video_urls.append(ep_id)
                downloaded_audio_urls.append(url)
        if stor_metadata.shape[0] > 0:
            playlist_csv = osp.join(episodes_dir, f'{playlist_id}.csv')
            refresh_metadata(playlist_csv, stor_metadata)

    # Save all metadata
    refresh_metadata(all_csv, stor_all_metadata)
