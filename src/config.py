import os
from dataclasses import dataclass
from urllib.parse import quote_plus

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class PostgresConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    database: str = os.getenv("POSTGRES_DB", "tutorial")

    @property
    def langchain_connection_string(self) -> str:
        user = quote_plus(self.user)
        password = quote_plus(self.password)
        database = quote_plus(self.database)

        return (
            f"postgresql+psycopg://{user}:{password}"
            f"@{self.host}:{self.port}/{database}"
        )

    @property
    def psycopg_connection_kwargs(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.database,
            "user": self.user,
            "password": self.password,
        }


@dataclass(frozen=True)
class VectorStoreConfig:
    table_name: str = os.getenv("VECTOR_TABLE_NAME", "books")
    vector_size: int = os.getenv("VECTOR_SIZE", 768)


postgres_config = PostgresConfig()
vector_store_config = VectorStoreConfig()