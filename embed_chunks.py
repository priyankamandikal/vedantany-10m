'''
Script to embed the full dataset into a vectorstore using a given embedding model
Run any of the following depending on the desired configuration:
    python embed_chunks.py --embedding_model nomic --vectorstore chroma
    python embed_chunks.py --embedding_model openai --vectorstore chroma
    python embed_chunks.py --embedding_model openai --vectorstore pinecone
'''
import time
import argparse
from torch.utils.data import DataLoader

from utils.setup import *
from create_chunks import ChunkDataset
from utils.init_components import init_vectorstore

parser = argparse.ArgumentParser()
# Chunking params
parser.add_argument("--whisper_model", type=str, default="large-v2", help="Whisper model to use. Options: large-v2")
parser.add_argument("--csv_file", type=str, default="bot.csv", help="CSV file containing metadata")
# Embedding params
parser.add_argument("--embedding_model", type=str, default="nomic", help="Embedding model to use. Options: openai, nomic")
parser.add_argument("--vectorstore", type=str, default="chroma", help="Vectorstore to use. Options: pinecone, chroma")
parser.add_argument("--index_name", type=str, default="vedantany-10m", help="Name of the index in vectorstore")
parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the output")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size for dataloader")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
args = parser.parse_args()


if __name__ == "__main__":

    if args.save_dir is None:
        args.save_dir = osp.join(EMBED_DIR, args.vectorstore, args.embedding_model, args.whisper_model, args.index_name)

    dataset = ChunkDataset(whisper_model=args.whisper_model,
                           csv_file=args.csv_file, 
                           create_chunks=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=10)

    print("\nEmbedding chunks...")
    vectorstore = init_vectorstore(args.vectorstore, args.embedding_model, args.whisper_model, args.index_name, create_db=True, persist_directory=args.save_dir, gpu_id=args.gpu_id)
    print("Number of batches:", len(dataloader))
    last_chunk = 0  # set accordingly
    for i, (split, metadata) in enumerate(dataloader):
        if i < last_chunk:
            continue
        print(f"Batch {i+1}/{len(dataloader)}")
        # convert metadata from dict of lists to list of dicts
        metadata = [{k: v[i] for k, v in metadata.items()} for i in range(len(metadata['ep_id']))]
        if args.embedding_model == 'nomic':
            # prepend "search_document" to the text
            split = [f"search_document: {text}" for text in split]
        start_time = time.time()
        vectorstore.add_texts(texts=split, metadatas=metadata)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        print("--------")
    print("... Done")