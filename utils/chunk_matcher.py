'''
Helper class to match passages to chunks and get the corresponding links. 
Used by embed_pinecone.py
'''

import numpy as np

class ChunkMatcher:
    def __init__(self, passages, chunks, links, searchlen=5, verbose=False, log_file="logs/chunk_matcher.log"):
        self.passages = passages
        self.chunks = chunks
        self.links = links
        self.searchlen = searchlen
        self.verbose = verbose
        self.log_file = log_file
    
        # Initialize sampled links
        N = len(self.passages)
        bins = np.linspace(0, len(self.links)-1, N, dtype=int)
        self.sampled_links = [self.links[i] for i in bins]

    @property
    def num_chunks(self):
        return len(self.chunks)

    def get_matching_links(self):
        # Iterate over passages and get matching chunks
        for idx, passage in enumerate(self.passages):
            matching_index = self.find_starting_chunk(passage)
            if matching_index is not None:
                self.sampled_links[idx] = self.links[matching_index]
        # Create embed links
        self.embed_links = self.sampled_links.copy()
        self.embed_links = [link.replace("watch?v=", "embed/").replace("&t=", "?start=") for link in self.embed_links]
        for link_idx, _ in enumerate(self.embed_links):
            if link_idx < len(self.embed_links) - 1:
                end_time = self.embed_links[link_idx + 1].split('?start=')[1]
                self.embed_links[link_idx] += f"&end={end_time}"
        return self.sampled_links, self.embed_links

    def find_starting_chunk(self, passage):
        passage_words = passage.split()

        # Iteration 0:
        # Find the first word in the passage in the chunks and create a filtered_chunks list of dict: chunk idx, search_idx, chunk len, chunk_count, chunk
        iter = 0
        search_word = passage_words[0]
        filtered_chunks = []
        for chunk_idx, chunk in enumerate(self.chunks):
            # find all idxs of search_word in chunk
            chunk_words = chunk.split()
            search_idxs = [i for i, word in enumerate(chunk_words) if word == search_word]
            if len(search_idxs) > 0:
                for search_idx in search_idxs:
                    filtered_chunks.append({'chunk_idx': chunk_idx, 'search_idx': search_idx, 'chunk_len': len(chunk_words), 'chunk_count': 1, 'chunk_text': chunk})
        if self.verbose:
            print(f"\n0 of {min(self.searchlen, len(passage_words))}: {search_word}")
            print(f"filtered_chunks: {filtered_chunks}")

        # Other iterations (max of searchlen):
        # for each filtered chunk, if search_idx + 1 > chunk_len, concatenate the next chunk to the current chunk. Update chunk_len.
        # if search_idx + 1 contains the next word in the passage, then continue. Update search_idx.
        # Else, remove the chunk from filtered_chunks

        # for iter in range(1, min(self.searchlen, len(passage_words))):
        while len(filtered_chunks) > 1 and iter < len(passage_words) - 1:
            iter += 1
            search_word = passage_words[iter]

            # Iterate over filtered_chunks
            for chunk in filtered_chunks.copy():
                chunk_idx = chunk['chunk_idx']
                search_idx = chunk['search_idx']
                chunk_len = chunk['chunk_len']
                chunk_count = chunk['chunk_count']
                chunk_text = chunk['chunk_text']

                # if search_idx + 1 > chunk_len, concatenate the next chunk to the current chunk. Update chunk_len.
                if search_idx + 1 == chunk_len:
                    if chunk_idx + chunk_count < self.num_chunks:
                        next_chunk = self.chunks[chunk_idx + chunk_count]
                        chunk_text += " " + next_chunk
                        chunk_len = len(chunk_text.split())
                        chunk['chunk_text'] = chunk_text
                        chunk['chunk_len'] = chunk_len
                        chunk['chunk_count'] = chunk_count + 1
                    else:
                        filtered_chunks.remove(chunk)
                        continue

                # if search_idx + 1 contains the next word in the passage, then continue. Update search_idx.
                if chunk_text.split()[search_idx + 1] == search_word:
                    chunk['search_idx'] = search_idx + 1
                else:
                    filtered_chunks.remove(chunk)

            if self.verbose:
                print(f"\n{iter+1} of {min(self.searchlen, len(passage_words) + 1)}: {search_word}")
                print(f"filtered_chunks: {filtered_chunks}")

        if len(filtered_chunks) == 0:
            with open(self.log_file, 'a') as f:
                video_link = self.sampled_links[0].split('&t=')[0]
                f.write(f"\n{video_link}\nmatches: {len(filtered_chunks)}\n{passage}\n")
            return None
        elif len(filtered_chunks) > 1:
            with open(self.log_file, 'a') as f:
                video_link = self.sampled_links[0].split('&t=')[0]
                f.write(f"\n{video_link}\nmatches: {len(filtered_chunks)}\n{passage}\n")
                f.write(f"chunk texts:\n")
                for chunk in filtered_chunks:
                    f.write(f"{chunk['chunk_text']}\n")
        return filtered_chunks[0]['chunk_idx']


if __name__=='__main__':

    # Test the function with a simple example
    passages = ['accomplish here?',
                'those questions. it will work then.']
    chunks = ['try to do it.',
              'what exactly are we trying to accomplish here?',
              'you should try to',
              'give answers to those questions.',
              'it will work then']
    links = ['https://www.youtube.com/watch?v=1&t=1',
             'https://www.youtube.com/watch?v=1&t=2',
             'https://www.youtube.com/watch?v=1&t=3',
             'https://www.youtube.com/watch?v=1&t=4',
             'https://www.youtube.com/watch?v=1&t=5']

    chunkmatcher = ChunkMatcher(passages, chunks, links, searchlen=5, verbose=True)
    sampled_links, embed_links = chunkmatcher.get_matching_links()
    print(f"\nSampled links: \n{sampled_links}")
    print(f"\nEmbed links: \n{embed_links}")