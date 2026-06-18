
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector


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
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="my_docs",
        connection="postgresql+psycopg://...",
    )
    
    print("Hello from scalable-books-search!")


if __name__ == "__main__":
    main()
