'''
Script to run the Ask Swami bot with a user query. 
It indexes into the saved Chroma VectorDB, retrieves the top K docs, and runs the Mixtral chatbot with the retreived sources. It then returns the answer and the top sources used to generate the answer.
Run as:
    python mixtral_bot.py --llm mixtral --embedding_model nomic --vectorstore chroma --k 5 --ensemble_k 100 --fusion_type similarity_fusion --query_indices 0 1 2 3 4
'''
import os
import os.path as osp
import pandas as pd
from time import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.tfidf_retriever import CustomTFIDFRetriever
from utils.ensemble_retriever import CustomEnsembleRetriever

from utils.init_components import init_vectorstore
from utils.bot_utils import *
from utils.setup import CHUNK_DIR


class MixtralBot:
    def __init__(self, fp16=False):
        self.model, self.tokenizer = self.load_mixtral(fp16)

    def load_mixtral(self, fp16=False):
        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        if fp16:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        return model, tokenizer
    
    def build_prompt(self, question, prompt_type="rag", retrievals=None, verbose=False):
        if "rag" in prompt_type:
            assert retrievals is not None
            return self.build_rag_prompt(question, retrievals, verbose)
        else:
            return self.build_standard_prompt(question, verbose)
        
    def build_rag_prompt(self, question, retrievals, verbose=False):
        retrievals_str = get_retrievals_str(retrievals)
        prompt = f"{RAG_PROMPT_PREFIX}\n\nSources:\n\n{retrievals_str}\nQuestion: {question}\n\nAnswer:"
        if verbose:
            print("\n", "="*30, "RAG prompt", "="*30)
            print(prompt)
        return prompt
    
    def build_standard_prompt(self, question, verbose=False):
        prompt = f"{STD_PROMPT_PREFIX}\n\nQuestion: {question}\n\nAnswer:"
        if verbose:
            print("\n", "="*30, "Standard prompt", "="*30)
            print(prompt)
        return prompt

    def without_chat_template(self, prompt, verbose=False):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        generated_ids = self.model.generate(**inputs, max_new_tokens=1000, pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if verbose:
            print("\n", "="*30, "Without chat template", "="*30)
            print(f"Prompt: \n{prompt}\n")
            print(answer)
        return answer

    def with_chat_template(self, prompt, verbose=False):
        chat = [
                {"role": "user", "content": prompt}
                ]
        inputs = self.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        templated_prompt = self.tokenizer.decode(inputs[0])
        generated_ids = self.model.generate(inputs, max_new_tokens=1000, pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        answer = answer[len(templated_prompt)-3:]
        if verbose:
            print("\n", "="*30, "With chat template", "="*30)
            print(f"Templated prompt: \n{templated_prompt}\n")
            print(answer)
        return answer
    
    def with_chat_template_batching(self, prompts, verbose=False):
        templated_prompts = []
        for prompt in prompts:
            chat = [
                    {"role": "user", "content": prompt}
                    ]
            templated_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            templated_prompts.append(templated_prompt)
        model_inputs = self.tokenizer(templated_prompts, return_tensors="pt", padding=True).to("cuda")
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=1000, pad_token_id=self.tokenizer.eos_token_id)
        answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for idx, templated_prompt in enumerate(templated_prompts):
            answers[idx] = answers[idx][len(templated_prompt)-3:]
        if verbose:
            print("\n", "="*30, "With chat template", "="*30)
            for templated_prompt, answer in zip(templated_prompts, answers):
                print(f"Templated Prompt: \n{templated_prompt}\n")
                print(answer)
        return answers


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="gpt-3.5-turbo", help="Name of llm model to use. Choose from [gpt-4, gpt-3.5-turbo, mixtral]")
    parser.add_argument("--embedding_model", type=str, default="nomic", help="Name of embedding to use. Choose from [openai, nomic]")
    parser.add_argument("--embedding_dir", type=str, default=None, help="Directory containing saved embeddings")
    parser.add_argument("--vectorstore", type=str, default="chroma", help="Name of vectorstore to use. Choose from [pinecone]")
    parser.add_argument("--whisper_model", type=str, default="large-v2", help="Whisper model to use. Options: large-v2")
    parser.add_argument("--index_name", type=str, default="ask-swami-bot", help="Name of the index in vectorstore")
    parser.add_argument("--k", type=int, default=5, help="Number of sources to retrieve")
    parser.add_argument("--ensemble_k", type=int, default=100, help="Number of sources to retrieve for ensemble retriever")
    parser.add_argument("--fusion_type", type=str, default="similarity_fusion", help="Fusion type for ensemble retriever. Choose from [similarity_fusion, rank_fusion]")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 for llm")
    parser.add_argument("--query_indices", type=int, nargs="+", default=[0], help="Indices of queries to run")
    parser
    args = parser.parse_args()

    K = args.k
    ENSEMBLE_K = args.ensemble_k
    WEIGHTS = [0.8, 0.2]
    KEYWORD_FILE = "eval/2-rag-vs-kwrag/keywords/aggregate.xlsx"
    OUTDIR = f"eval/2-rag-vs-kwrag/answers/{args.llm}-{args.embedding_model}"
    JSON_DIR = osp.join(OUTDIR, f"json")
    HTML_DIR = osp.join(OUTDIR, f"html")
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(HTML_DIR, exist_ok=True)

    # intialize vectorstore and llm
    vectorstore = init_vectorstore(args.vectorstore, args.embedding_model, args.whisper_model, args.index_name, create_db=False)
    llm = MixtralBot(args.fp16)

    # read the documents from all the json files in chunk_dir
    chunk_dir = osp.join(CHUNK_DIR, args.whisper_model)
    texts_all = []
    metadatas_all = []
    for file in os.listdir(chunk_dir):
        with open(osp.join(chunk_dir, file), 'r') as f:
            data = json.load(f)
            for ele in data:
                texts_all.append(ele['text'])
                metadatas_all.append(ele['metadata'])

    # deep retriever
    deep_retriever = vectorstore.as_retriever(search_kwargs={'k': args.ensemble_k})

    # tfidf retriever
    tfidf_retriever = CustomTFIDFRetriever.from_texts(
        texts=texts_all, 
        metadatas=metadatas_all, 
        k=args.ensemble_k
    )

    # ensemble retriever with query search for tfidf
    ensemble_retriever_nokw = CustomEnsembleRetriever(
        retrievers=[tfidf_retriever, deep_retriever],
        weights=WEIGHTS,
        includes_nomic=args.embedding_model == "nomic",
        use_keywords=False
    )

    # ensemble retriever with keyword search for tfidf
    ensemble_retriever_kw = CustomEnsembleRetriever(
        retrievers=[tfidf_retriever, deep_retriever],
        weights=WEIGHTS,
        includes_nomic=args.embedding_model == "nomic",
        use_keywords=True
    )

    # read queries
    df = pd.read_excel(KEYWORD_FILE)
    categories = df['Category'].tolist()
    queries = df['Query'].tolist()
    keywords = df['Keywords'].tolist()
    print(f"Number of queries: {len(queries)}")

    # run the bot
    prompt_types = ["rag-kw", "rag"]
    prompt_ids = ["A", "B"]
    ER_obj = ExpandRetrievals(chunk_dir, tfidf_score_thr=0.1, sentence_split_n=1)
    for idx, (c, q, k) in enumerate(zip(categories, queries, keywords)):
        start_time = time()
        if idx not in args.query_indices:
            continue
        print("="*80)
        print(f"Processing {idx}/{len(queries)}...")
        print(f"Category: {c}")
        print(f"Query: {q}")
        print(f"Keywords: {k}")
        # query the retrievers
        deep_docs = get_docs(deep_retriever, q, args.embedding_model)[:K]
        tfidf_docs = get_docs(tfidf_retriever, q, args.embedding_model)[:K]
        ensemble_args = {"k": k, "fusion_type": args.fusion_type, "ensemble_k": ENSEMBLE_K}
        ensemble_docs_kw = get_docs(ensemble_retriever_kw, q, args.embedding_model, ensemble_args)[:K]
        ensemble_docs_nokw = get_docs(ensemble_retriever_nokw, q, args.embedding_model, ensemble_args)[:K]
        prompts = []
        retrievals = []
        for prompt_type, prompt_id in zip(prompt_types, prompt_ids):
            # extract the retrievals
            if prompt_type == "standard":
                r = None
            elif prompt_type == "rag":
                r = extract_retrievals(deep_docs, args.embedding_model)
                if args.k == 1:
                    r = [r[0]]
                else:
                    r = r[:args.k]
            elif prompt_type == "rag-kw":
                r = extract_retrievals(ensemble_docs_kw, args.embedding_model, tfidf_score_thr=0.1)
                r = ER_obj.expand_retrievals(r, k, tfidf_docs)
            retrievals.append(r)
            # print(f"Extracted retrievals")
            
            # build the prompt
            prompt = llm.build_prompt(q, prompt_type, retrievals=r, verbose=False)
            prompts.append(prompt)

        # run the llm
        answers = llm.with_chat_template_batching(prompts, verbose=False)
        print(f"Generated answers")
        # answers = ["Test"]*len(prompt_types)

        # save json and html
        fname = f"{c}_{q.replace(' ', '_')[:50]}"
        fname = "".join([c for c in fname if c.isalnum() or c in ['_', ' ', '-']])
        save_json(q, c, k, answers, retrievals, prompt_types, prompt_ids, fname, JSON_DIR)
        save_html(q, c, None, prompt_types, fname, JSON_DIR, HTML_DIR)

        print(f"Time taken: {(time()-start_time)/60:.2f} mins")
        # break