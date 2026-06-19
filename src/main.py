import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_postgres import PGVector   # old version of PGVector

from src.utils import table_exists, load_pdf_pages, establish_vector_store_connection, get_all_pdf_names, init_vector_store
embeddings = HuggingFaceEmbeddings(                             # TODO: consider using gemini 
    # TODO: how can we load the model as one of the files to avoid reloading it form HuggingFace?
    model_name="sentence-transformers/all-mpnet-base-v2",       # TODO: consider using other models?
    encode_kwargs={"normalize_embeddings": True},
)

def get_snippets(query, embeddings=embeddings, k=5,):
    pg_engine, vector_store = establish_vector_store_connection(embeddings)
    results = vector_store.similarity_search_with_score(query, k=k)
    for r in results:
        doc, score = r
        print(f"Score: {score}\n")
        print(doc)
    return results
    
def main():
    # Some code is adapted from 
    # https://docs.langchain.com/oss/python/langchain/knowledge-base#huggingface

    # TODO: Do we want to have Langsmith?

    # To load all books run this
    # init_vector_store(embeddings=embeddings)

    # To connect run this
    pg_engine, vector_store = establish_vector_store_connection(embeddings)
    results = vector_store.similarity_search_with_score("How to prevent overload?", k=5)
    for r in results:
        doc, score = r
        print(f"Score: {score}\n")
        print(doc)


    print("Hello from scalable-books-search!")


if __name__ == "__main__":
    main()
