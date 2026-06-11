The general idea of the project is to create scalable vectore store for (for example current news) and find related topics to queries.

Components:

RSS ingestion                               # partially done
    ↓
LangChain text splitter                     # texts are too small, not relevant yet
    ↓
LangChain embedding wrapper                 # done
    ↓
PostgreSQL + pgvector                       # todo: watch the YT tutorial for 40 mins
    ↓
Backend API semantic search endpoint        # not started (Langchain retriever?)
    ↓
Frontend                                    # not started