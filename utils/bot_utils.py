import os
import os.path as osp
import json
import spacy
from spacy.lang.en import English
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAG_PROMPT_PREFIX = "You are a helpful assistant that accurately answers queries using Swami Sarvapriyananda's YouTube talks. Use relevant information in the following passages to provide a detailed answer to the user query."

STD_PROMPT_PREFIX = "You are a helpful assistant that accurately answers queries using Swami Sarvapriyananda's YouTube talks. Provide a detailed answer to the user query."

class ExpandRetrievals:

    def __init__(self, chunk_dir, tfidf_score_thr=0.1, sentence_split_n=1, min_sentences=5, pad_left=2, pad_right=5, break_every_n=25, character_split_n=50, max_char_sentence_ratio=100, min_snippet_len=1200):
        self.chunk_dir = chunk_dir
        self.tfidf_score_thr = tfidf_score_thr
        self.sentence_split_n = sentence_split_n
        self.min_sentences = min_sentences
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.break_every_n = break_every_n
        self.max_char_sentence_ratio = max_char_sentence_ratio
        self.min_snippet_len = min_snippet_len
        self.sentencizer = English()
        self.sentencizer.add_pipe("sentencizer")
        self.character_splitter = RecursiveCharacterTextSplitter(chunk_size=character_split_n, 
                                                           chunk_overlap=0)
        self.stop_words = self.get_stop_words()
        
    def get_stop_words(self):
        # read stop words from file
        with open("eval/stopwords-nltk.txt", "r") as f:
            stopwords_nltk = f.read().splitlines()
        with open("eval/stopwords-spacy.txt", "r") as f:
            stopwords_spacy = f.read().splitlines()
        # union of NLTK and spaCy stop words
        stop_words = list(set(stopwords_spacy).union(set(stopwords_nltk)))
        stop_words.extend(["swami", "swamiji", "swamis", "swamijis", "vedanta", "upanishad", "upanishads"])
        return stop_words
    
    def get_snippet(self, text, keywords, pad_left=25, pad_right=100):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        occurrences = []
        for keyword in keywords:
            occurrences.extend([token.i for token in doc if token.text.lower() == keyword.lower()])
        if occurrences == []:
            return text
        min_occurrence = min(occurrences)
        max_occurrence = max(occurrences)
        # pad on either side by as many words as specified
        possible_starts = [i for i in range(min_occurrence - pad_left, min_occurrence + 1) if i >= 0]
        start = min(possible_starts)
        possible_ends = [i for i in range(max_occurrence, max_occurrence + pad_right + 1) if i < len(doc)]
        end = max(possible_ends)
        main_text = doc[start:end].text
        prev_text = doc[:start].text
        next_text = doc[end:].text
        return main_text, prev_text, next_text
    
    def pad_snippet(self, main_text, prev_text, next_text, pad_left=5, pad_right=5):
        prev_sentences = [sent.text for sent in self.sentencizer(prev_text).sents]
        next_sentences = [sent.text for sent in self.sentencizer(next_text).sents]
        # pad on either side by as many sentences as specified
        possible_starts = [i for i in range(len(prev_sentences) - pad_left, len(prev_sentences) + 1) if i >= 0]
        start = min(possible_starts)
        possible_ends = [i for i in range(0, pad_right + 1) if i <= len(next_sentences)]
        end = max(possible_ends)
        prev_snippet = " ".join(prev_sentences[start:])
        next_snippet = " ".join(next_sentences[:end])
        snippet = prev_snippet + " " + main_text + " " + next_snippet
        print(prev_snippet)
        print(main_text)
        print(next_snippet)
        return snippet
    
    def add_linebreaks_to_snippet(self, text, every_n=10):
        doc = self.sentencizer(text)
        sentences = [sent.text for sent in doc.sents]
        if len(sentences) < every_n:
            # use character splitter
            sentences = self.character_splitter.split_text(text)
        snippet = ""
        for i, sent in enumerate(sentences):
            snippet += sent + " "
            if i % every_n == 0 and i != 0:
                snippet += "\n\n"
        return snippet

    def expand_retrievals(self, retrievals, keywords, tfidf_docs):
        tfidf_doc_links = [doc.metadata["link"] for doc in tfidf_docs]
        kw_list = keywords.split(", ")
        kw_words = []
        for kw in kw_list:
            kw_words.extend(kw.split(" "))
        kw_words = set(kw_words)
        kw_words = kw_words.difference(self.stop_words)
        for r_idx, retrieval in enumerate(retrievals):
            # print("Current similarity score:", retrieval["sim_score_dict"])
            ep_id = retrieval["ep_id"]
            expanded_content = retrieval["content"]
            # get metadata from matching json file
            with open(osp.join(self.chunk_dir, f"{ep_id}.json"), 'r') as f:
                data = json.load(f)
                # get index of the retrieval in the json file with keyword match
                for i, ele in enumerate(data):
                    if ele["metadata"]["link"] == retrieval["link"]:
                        # previous
                        if i > 0:
                            prev_link = data[i - 1]["metadata"]["link"]
                            expanded_content = data[i - 1]["text"] + " " + expanded_content
                            retrieval["link"] = prev_link
                        # next
                        if i < len(data) - 1:
                            expanded_content = expanded_content + " " + data[i + 1]["text"]
                        break
                    
            # split the content into sets each containing n sentences
            doc = self.sentencizer(expanded_content)
            sents = [sent.text for sent in doc.sents]
            retrieval["needs_formatting"] = False
            chars_sent_ratio = len(expanded_content) / len(sents)
            print("Chars per sentence ratio:", chars_sent_ratio)
            if chars_sent_ratio > self.max_char_sentence_ratio:
                print("Using character splitter")
                # use character splitter
                sents = self.character_splitter.split_text(expanded_content)
                retrieval["needs_formatting"] = True
            sent_list = []
            unique_kw_set = set()
            kw_present_list = []
            for i in range(0, len(sents), self.sentence_split_n):
                sent_comb = " ".join(sents[i:i+self.sentence_split_n])
                sent_list.append(sent_comb)
                # remove punctuation
                sent_comb_nopunct = sent_comb.replace("'s", "")
                sent_comb_nopunct = sent_comb_nopunct.replace("' ", " ")
                sent_comb_nopunct = "".join([char for char in sent_comb_nopunct if (char.isalnum() or char.isspace() or char=="-")])
                sent_comb_words = set(sent_comb_nopunct.lower().split(" "))
                # keyword match
                if any(kw in sent_comb_words for kw in kw_words):
                    kw_present_list.append(True)
                    unique_kw_set = unique_kw_set.union(sent_comb_words.intersection(kw_words))
                else:
                    kw_present_list.append(False)
            # content will be the sentences from the first True keyword match till the last True keyword match
            if True in kw_present_list:
                start_i = kw_present_list.index(True)
                end_i = len(kw_present_list) - 1 - kw_present_list[::-1].index(True)
                retrieval["content"] = " ".join(sent_list[start_i:end_i+1])
                retrieval["sent_list"] = sent_list
                retrieval["start_i"] = start_i
                retrieval["end_i"] = end_i
                retrieval["unique_kw"] = list(unique_kw_set)
            else:
                retrieval["unique_kw"] = []

            retrievals[r_idx] = retrieval

        # get all retrievals with max number of unique keywords
        max_num_kw = max([len(retrieval["unique_kw"]) for retrieval in retrievals])
        r_idxs = [i for i, retrieval in enumerate(retrievals) if len(retrieval["unique_kw"]) == max_num_kw]
        # if any keyword is present in the title, update r_idxs
        r_idxs_title = [i for i in r_idxs if any(kw in [word.lower() for word in retrievals[i]["title"].split(" ")] for kw in kw_words)]
        if r_idxs_title:
            r_idxs = r_idxs_title
        # select the one with the longest content
        max_len = max([len(retrieval["content"]) for retrieval in [retrievals[i] for i in r_idxs]])
        r_idx = [i for i in r_idxs if len(retrievals[i]["content"]) == max_len][0]

        # pad the snippet
        start_i = retrievals[r_idx]["start_i"]
        end_i = retrievals[r_idx]["end_i"]
        sent_list = retrievals[r_idx]["sent_list"]
        pad = True
        while pad or len(" ".join(sent_list[start_i:end_i+1])) < self.min_snippet_len:
            print(pad, len(" ".join(sent_list[start_i:end_i+1])))
            pad = False
            if self.pad_left:
                possible_starts = [i for i in range(start_i - self.pad_left, start_i + 1) if i >= 0]
                start_i = min(possible_starts)
            if self.pad_right:
                possible_ends = [i for i in range(end_i, end_i + self.pad_right + 1) if i < len(sent_list)]
                end_i = max(possible_ends)
        snipped_sent_list = sent_list[start_i:end_i+1]
        print("Number of sentences in chosen snippet:", len(snipped_sent_list))
        # add line breaks
        for i in range(len(snipped_sent_list)):
            if i % self.break_every_n == 0 and i != 0:
                # ensure that last few sentences are not too short
                if len(snipped_sent_list) - i > self.break_every_n//3:
                    print("Adding line break")
                    snipped_sent_list[i] += "\n\n"
        retrievals[r_idx]["content"] = " ".join(snipped_sent_list)
        return [retrievals[r_idx]]

def extract_retrievals(docs, embedding_model, tfidf_score_thr=0.1):
    retrievals = []
    for doc in docs:
        doc_dict = {}
        if type(doc) == tuple:
            doc_dict["similarity"] = doc[1].copy()
            doc = doc[0]
        doc_dict["title"] = doc.metadata['title']
        doc_dict["link"] = doc.metadata['link']
        doc_dict['ep_id'] = doc.metadata['ep_id']
        if "sim_score_dict" in doc.metadata:
            doc_dict["sim_score_dict"] = doc.metadata["sim_score_dict"]
            if doc.metadata["sim_score_dict"][0] < tfidf_score_thr:
                print("Skipping retrieval with low similarity score")
                continue
        if embedding_model == "nomic":
            if doc.page_content.startswith("search_document: "):
                doc_dict["content"] = doc.page_content[len("search_document: "):]
            else:
                doc_dict["content"] = doc.page_content
        else:
            doc_dict["content"] = doc.page_content
        retrievals.append(doc_dict)
    return retrievals

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def print_result(result):
    print("\nAnswer:\n\n", result['answer'])
    print("\nSource:\n")
    for doc in result['context']:
        print(doc.metadata['title'])
        print(doc.page_content, "\n")

def get_docs(retriever, q, EMBEDDING_MODEL, ensemble_args=None):
    if EMBEDDING_MODEL == "nomic":
        nomic_q = f"search_query: {q}"
        docs = retriever.get_relevant_documents(nomic_q, ensemble_args=ensemble_args)
    else:
        docs = retriever.get_relevant_documents(q, ensemble_args=ensemble_args)
    return docs

def get_retrievals_str(retrievals, add_title=True, verbose=False):
    retrievals_str = ""
    for retrieval in retrievals:
        if add_title:
            retrievals_str += f"Lecture: {retrieval['title']}\n"
        retrievals_str += f"Passage: {retrieval['content']}\n"
        retrievals_str += "\n"
    if verbose:
        print(retrievals_str)
    return retrievals_str

def get_response_html(q=None, c=None, k=None, a=None, r=None, model_name=None):
    html_str = ""
    if c:
        html_str += f"<h1>Category: {c}</h1>"
    if q:
        html_str += f"<h2>Query: {q}</h2>"
    if k:
        html_str += f"<h2>Keywords: {k}</h2>"
    if a:
        if model_name:
            html_str += f"<h2>Model {model_name}</h2>"
        else:
            html_str += f"<h2>Answer:</h2>"
        a = a.replace('\n', '<br>')
        html_str += f"<p style='text-align: justify;'>{a}</p>"
    if r:
        html_str += f"<h2>Source:</h2>"
        for retrieval in r:
            html_str += f"<h3>{retrieval['title']}</h3>"
            retrieval['content'] = retrieval['content'].replace('\n', '<br>')
            html_str += f"<p style='text-align: justify;'>{retrieval['content']}</p>"
    return html_str

def get_combined_html(q, c, json_files, k=None):
    html_str = ""
    with open("utils/prefix.html", "r") as f:
        html_str += f.read()
    html_str += "\n<body>\n"
    html_str += f"<h1>Category: {c}</h1>"
    html_str += f"<h2>Query: {q}</h2>"
    if k:
        html_str += f"<h2>Keywords: {k}</h2>"
    html_str += "\n<comparison>\n"
    for json_file in json_files:
        html_str += '<div class="text-sample">'
        with open(json_file, "r") as f:
            data = json.load(f)
            html_str += get_response_html(a=data['a'], r=data['r'], model_name=data['prompt_id'])
        html_str += '</div>'
    html_str += "\n</comparison>\n</body>\n</html>"
    return html_str

def save_json(q, c, k, answers, retrievals, prompt_types, prompt_ids, fname, json_dir):
    for a, r, prompt_type, prompt_id in zip(answers, retrievals, prompt_types, prompt_ids):
        outdir = osp.join(json_dir, prompt_type)
        os.makedirs(outdir, exist_ok=True)
        outfile = osp.join(outdir, f"{fname}.json")
        with open(outfile, "w") as f:
            dictionary = {"q": q, "c": c, "k": k, "a": a, "r": r, 
                        "prompt_type": prompt_type,
                        "prompt_id": prompt_id
                        }
            json_object = json.dumps(dictionary, indent=4)
            f.write(json_object)

def save_html(q, c, k, prompt_types, fname, json_dir, html_dir):
    json_files = [osp.join(json_dir, prompt_type, f"{fname}.json") for prompt_type in prompt_types]
    html_str = get_combined_html(q, c, json_files, k)
    with open(osp.join(html_dir, f"{fname}.html"), "wb") as f:
        f.write(html_str.encode("utf-8"))
