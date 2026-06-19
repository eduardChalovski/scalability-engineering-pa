import pypdf
from langchain_core.documents import Document

import psycopg
from src.config import postgres_config

from langchain_postgres import PGEngine, PGVectorStore
from src.config import postgres_config, vector_store_config


def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_path, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]


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
        

def establish_vectore_store_connection(embeddings):
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