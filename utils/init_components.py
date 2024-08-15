"""
Helper functions to initialize different components for the RAG pipeline
Used by: embed_chunks.py, run_bot.py
"""

import os
import os.path as osp
import shutil

from utils.setup import EMBED_DIR

def init_embedding(embedding_name, gpu_id=0):
    if 'openai' in embedding_name:
        from langchain_openai import OpenAIEmbeddings
        embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    elif 'nomic' in embedding_name:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedding = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", 
                                          model_kwargs={"trust_remote_code": True,
                                                        "device": f"cuda:{gpu_id}"})
    else:
        raise ValueError("Invalid embedding name")
    return embedding

def init_vectorstore(vectorstore_name, embedding_model, whisper_model, index_name, create_db=False, persist_directory=None, gpu_id=0):
    print(f"\nInitializing vectorstore {vectorstore_name} using {embedding_model}...")
    embedding = init_embedding(embedding_model, gpu_id)
    if vectorstore_name == 'pinecone':
        vectorstore = _init_pinecone(index_name, embedding, create_db)
    elif vectorstore_name == 'chroma':
        if persist_directory is None:
            persist_directory = osp.join(EMBED_DIR, vectorstore_name, embedding_model, whisper_model, index_name)
        print(f"Persist directory: {persist_directory}")
        vectorstore = _init_chroma(index_name, embedding, create_db, persist_directory)
    else:
        raise ValueError("Invalid vectorstore name")
    return vectorstore

def _init_pinecone(index_name, embedding, create_db=False):
    import pinecone
    from langchain_community.vectorstores.pinecone import Pinecone
    pinecone.init(
        api_key=os.environ.get('PINECONE_API_KEY'),  
        environment=os.environ.get('PINECONE_ENV', 'us-west1-gcp-free')
    )
    print(f"Loading existing index {index_name}...")
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embedding)
    return vectorstore

def _init_chroma(index_name, embedding, create_db=False, persist_directory=None):
    from langchain_community.vectorstores.chroma import Chroma
    if create_db:
        print(f"Creating new vectorstore {index_name}")
        if osp.exists(persist_directory):
            shutil.rmtree(persist_directory)
        os.makedirs(persist_directory)
    vectorstore = Chroma(embedding_function=embedding,
                         collection_name=index_name,
                         persist_directory=persist_directory
                         )
    return vectorstore

def init_llm(llm_name, temperature=0, max_new_tokens=10000):
    if 'gpt' in llm_name:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name=llm_name, temperature=temperature)
    else:
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
        llm = HuggingFacePipeline(pipeline=pipe)
    return llm
