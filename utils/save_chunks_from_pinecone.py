import os
import numpy as np
import json
import pinecone

from utils.setup import *

model_name = "large-v2"
chunk_dir = osp.join(CHUNK_DIR, model_name)
os.makedirs(chunk_dir, exist_ok=True)

def get_ids_from_query(index,input_vector):
    print("searching pinecone...")
    results = index.query(
    top_k=10000,
    include_values=False,
    include_metadata=False,
    vector=input_vector,
    )
    ids = set()
    for result in results['matches']:
        ids.add(result.id)
    return ids

def get_all_ids_from_index(index, num_dimensions, namespace=""):
    num_vectors = index.describe_index_stats()
    num_vectors = num_vectors.namespaces[namespace].vector_count
    all_ids = set()
    while len(all_ids) < num_vectors:
        print("Length of ids list is shorter than the number of total vectors...")
        input_vector = np.random.rand(num_dimensions).tolist()
        print("creating random vector...")
        ids = get_ids_from_query(index,input_vector)
        print("getting ids from a vector query...")
        all_ids.update(ids)
        print("updating ids set...")
        print(f"Collected {len(all_ids)} ids out of {num_vectors}.")
    print("Collected all ids.")
    return all_ids

def save_chunks_from_id(index, ids):
    print("saving chunks...")
    results = index.fetch(ids)
    for key, result in results['vectors'].items():
        metadata = result['metadata']
        link = metadata['link']
        ep_id = link.split("v=")[1].split("&")[0]
        timestamp = int(link.split("&t=")[1])
        metadata['ep_id'] = ep_id
        metadata['timestamp'] = timestamp
        metadata['id'] = key
        text = metadata['text']
        metadata.pop('text')
        if os.path.exists(osp.join(chunk_dir, f'{ep_id}.json')):
            with open(osp.join(chunk_dir, f'{ep_id}.json'), 'r') as f:
                data = json.load(f)
        else:
            data = []

        with open(osp.join(chunk_dir, f'{ep_id}.json'), 'w') as f:
            position = 0
            for i, entity in enumerate(data):
                if entity['metadata']['timestamp'] > timestamp:
                    position = i
                    break
            else:
                position = len(data)
            data.insert(position, {'text': text, 'metadata': metadata})
            json.dump(data, f)


if __name__ == "__main__":

    # initialize pinecone index
    pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),  
    environment=os.environ.get('PINECONE_ENV', 'us-west1-gcp-free')
    )
    pinecone_index = pinecone.Index("vedantany-10m")
    dimension = 1536  # dim of text-embedding-ada-002

    # get all ids from the index
    all_ids = get_all_ids_from_index(pinecone_index, num_dimensions=dimension)
    all_ids = list(all_ids)
    with open(osp.join(metadata_dir, model_name,  "pinecone_ids.txt"), "w") as f:
        for id in all_ids:
            f.write(id + "\n")
    
    # save chunks from the ids
    for i in range(0, len(all_ids), 1000):
        k = min(i + 1000, len(all_ids))
        save_chunks_from_id(pinecone_index, all_ids[i:k])


