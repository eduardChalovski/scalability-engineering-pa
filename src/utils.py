import pypdf
from langchain_core.documents import Document

import psycopg
from src.config import postgres_config

from langchain_postgres import PGEngine, PGVectorStore
from src.config import postgres_config, vector_store_config

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Some code is adapted from 
# https://docs.langchain.com/oss/python/langchain/knowledge-base#huggingface

from langchain_core.documents import Document

def clean_text(text: str) -> str:
    return (
        text
        .replace("\x00", "")
        .replace("\ufeff", "")
        .strip()
    )


def clean_document(document: Document) -> Document:
    return Document(
        page_content=clean_text(document.page_content),
        metadata=document.metadata,
    )


def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]


def get_all_pdf_names(directory_name="books_for_semantic_search"):
    """
    Iterates over files in directory_name and returns pdf names.

    Args:
        directory_name:     Name of the directory to search for pdfs.

    Returns:
        pdf_names:          List of pdf file names.

    Raises:
        KeyError: Right now doesn't raise any exceptions. TODO: introduce some checks
    """
    my_path = Path(directory_name)

    pdf_names = [
        str(file)
        for file in my_path.iterdir()
        if file.is_file() and file.name.endswith(".pdf")
    ]

    return pdf_names


def table_exists(
    table_name: str,
    schema: str = "public",
) -> bool:
    with psycopg.connect(**postgres_config.psycopg_connection_kwargs) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass(%s) IS NOT NULL;",
                (f"{schema}.{table_name}",),
            )
            result = cur.fetchone()
            return bool(result[0])
        

def establish_vector_store_connection(embeddings):
    """
    Establish connection with the PostgreSQL docker container. Creates the vectore store if it doesn't exist.

    Args:
        embeddings:     Model used for embeddings e.g.HuggingFaceEmbeddings

    Returns:
        pg_engine:      An instance for managing connections to a Postgres database.
        vector_store:   Postgres Vector Store instance.

    Raises:
        KeyError: Right now doesn't raise any exceptions. TODO: introduce some checks
    """
    
    pg_engine = PGEngine.from_connection_string(
        url=postgres_config.langchain_connection_string
    )

    if not table_exists(vector_store_config.table_name):
        print(f"Creating vector table: {vector_store_config.table_name}")

        pg_engine.init_vectorstore_table(
            table_name=vector_store_config.table_name,
            vector_size=vector_store_config.vector_size,
        )
    else:
        print(f"Vector table already exists: {vector_store_config.table_name}")

    vector_store = PGVectorStore.create_sync(
        engine=pg_engine,
        table_name=vector_store_config.table_name,
        embedding_service=embeddings
    )

    return pg_engine, vector_store

def init_vector_store(embeddings, directory_name: str = None):
    # Establish connection to the vectore store.
    pg_engine, vector_store = establish_vector_store_connection(embeddings)

    if directory_name is None:
        pdfs_file_paths = get_all_pdf_names()
    else:
        pdfs_file_paths = get_all_pdf_names(directory_name)

    docs = []

    for pdf_path in pdfs_file_paths:
        loaded_pages = load_pdf_pages(pdf_path)
        cleaned_pages = [clean_document(doc) for doc in loaded_pages]
        docs.extend(cleaned_pages)

    print("pages added : ", len(docs))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    ids = vector_store.add_documents(documents=all_splits)


    print("splits from the pages added: ", len(all_splits))