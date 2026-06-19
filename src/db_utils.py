import pypdf
from langchain_core.documents import Document

import psycopg
from src.config import postgres_config

# Below is a minimal helper for demonstration purposes.
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