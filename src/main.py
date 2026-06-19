import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_postgres import PGVector   # old version of PGVector

from src.db_utils import table_exists, load_pdf_pages, establish_vectore_store_connection

def main():
    # Some code is adapted from 
    # https://docs.langchain.com/oss/python/langchain/knowledge-base#huggingface

    # TODO: Do we want to have Langsmith?

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",       # TODO: consider using other models?
        encode_kwargs={"normalize_embeddings": True},
    )

    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]

    vector_1 = embeddings.embed_query(documents[0].page_content)
    vector_2 = embeddings.embed_query(documents[1].page_content)

    assert len(vector_1) == len(vector_2)
    print(f"Generated vectors of length {len(vector_1)}\n")
    print(vector_1[:10])
    
    # How to set up the connection to the postgresdb
    # Python running on Windows       -> use localhost
    # Python running in Docker Compose -> use postgres
    # pgAdmin running in Docker        -> use postgres
    # psql from Windows                -> use localhost

    # CONNECTION_STRING = postgres_config.langchain_connection_string
    # COLLECTION_NAME = "my_docs"
    # VECTOR_TABLE_NAME = vector_store_config.table_name
    # VECTOR_SIZE=vector_store_config.vectore_size

    # Establish connection to the vectore store.
    pg_engine, vectore_store = establish_vectore_store_connection(embeddings)

    
    # file_path = "../example_data/nke-10k-2023.pdf"
    file_path = "./books_for_semantic_search/Distributed_Systems_4.pdf"
    
    docs = load_pdf_pages(file_path)
    print("pages added : ", len(docs))

    # Old working version, apparently PGVectorStore is the new gen of PGVector
    # vector_store = PGVector(
    #     embeddings=embeddings,
    #     collection_name="my_docs",
    #     connection="postgresql+psycopg://postgres:postgres@localhost:5432/tutorial",
    # )
    
    print("Hello from scalable-books-search!")


if __name__ == "__main__":
    main()
