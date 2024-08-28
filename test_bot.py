'''
Script to run the chatbot with a user query. 
It indexes into the saved VectorDB, retrieves the top K docs, and prompts the llm with the retreived sources. It then returns the top answer and the sources used to generate the answer.

Run any of the following according to the model and embedding to use:
    python test_bot.py --llm gpt-4 --embedding_model openai --vectorstore pinecone
    python test_bot.py --llm gpt-3.5-turbo --embedding_model openai --vectorstore pinecone
    python test_bot.py --llm gpt-3.5-turbo --embedding_model nomic --vectorstore chroma
    python test_bot.py --llm mixtral --embedding_model nomic --vectorstore chroma

To pass an input query at run time:
    python test_bot.py --llm mixtral --embedding_model nomic --vectorstore chroma --pass-query
'''
import argparse
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from utils.init_components import init_vectorstore, init_llm

parser = argparse.ArgumentParser()
parser.add_argument("--llm", type=str, default="gpt-3.5-turbo", help="Name of llm model to use. Choose from [gpt-4, gpt-3.5-turbo, mixtral]")
parser.add_argument("--embedding_model", type=str, default="nomic", help="Name of embedding to use. Choose from [openai, nomic]")
parser.add_argument("--embedding_dir", type=str, default=None, help="Directory containing saved embeddings")
parser.add_argument("--vectorstore", type=str, default="chroma", help="Name of vectorstore to use. Choose from [pinecone]")
parser.add_argument("--whisper_model", type=str, default="large-v2", help="Whisper model to use. Options: large-v2")
parser.add_argument("--index_name", type=str, default="vedantany-10m", help="Name of the index in vectorstore")
parser.add_argument("--pass-query", action="store_true", help="Pass a query to the chatbot")
args = parser.parse_args()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":

    # Input query and k
    if args.pass_query:
        QUERY = input("\nEnter query: ")
        while True:
            try:
                K = int(input("\nEnter number of sources to retrieve: "))
                break
            except ValueError:
                print("Enter an integer")
    else:
        QUERY = "What is Brahman?"
        K = 1
        print("\nQuery: ", QUERY)

    # intialize retriever and llm
    vectorstore = init_vectorstore(args.vectorstore, args.embedding_model, args.whisper_model, args.index_name, create_db=False, persist_directory=args.embedding_dir)
    retriever = vectorstore.as_retriever(search_kwargs={'k': K})
    llm = init_llm(args.llm, temperature=0)

    # prompt template
    template = """You are a helpful assistant that accurately answers queries using Swami Sarvapriyananda's YouTube talks. Use the following passages to provide a detailed answer to the user query.

    {context}

    Question: {question}

    Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    # build the chain
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # run the chain
    result = rag_chain_with_source.invoke(QUERY)
    print("\nAnswer:\n\n", result['answer'])
    print("\nSources:\n")
    for doc in result['context']:
        print(doc.page_content, "\n")
