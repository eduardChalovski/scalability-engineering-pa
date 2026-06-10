# -- Source - https://stackoverflow.com/a/75249163
# -- Posted by Indra Dwi Aryadi
# -- Retrieved 2026-06-10, License - CC BY-SA 4.0

import feedparser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# rss_url = 'https://feeds.bbci.co.uk/news/politics/rss.xml?edition=uk'

rss_feeds = [
    {"source": "BBC Top Stories", "url": "https://feeds.bbci.co.uk/news/rss.xml"},
    # {"source": "BBC World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    # {"source": "BBC Technology", "url": "https://feeds.bbci.co.uk/news/technology/rss.xml"},
    # {"source": "BBC Business", "url": "https://feeds.bbci.co.uk/news/business/rss.xml"},
    # {"source": "BBC Science & Environment", "url": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"},

    # {"source": "Guardian World", "url": "https://www.theguardian.com/world/rss"},
    # {"source": "Guardian Technology", "url": "https://www.theguardian.com/technology/rss"},
    # {"source": "Guardian Business", "url": "https://www.theguardian.com/business/rss"},
    # {"source": "Guardian Science", "url": "https://www.theguardian.com/science/rss"},
    # {"source": "Guardian Environment", "url": "https://www.theguardian.com/environment/rss"},

    # {"source": "NPR News", "url": "https://feeds.npr.org/1001/rss.xml"},
    # {"source": "NPR World", "url": "https://feeds.npr.org/1004/rss.xml"},
    # {"source": "NPR Technology", "url": "https://feeds.npr.org/1019/rss.xml"},

    # {"source": "Tagesschau Top", "url": "https://www.tagesschau.de/xml/rss2"},
    # {"source": "Tagesschau Ausland", "url": "https://www.tagesschau.de/ausland/index~rss2.xml"},
    # {"source": "Tagesschau Wirtschaft", "url": "https://www.tagesschau.de/wirtschaft/index~rss2.xml"},
]



def main():
    log_dict = {}
    failed_rss = []
    documents = []
    number_of_entries_to_add = 2
    log_dict["[PLANNED 1]"] = f"ADD {number_of_entries_to_add} RSS entries"
    for rss_feed in rss_feeds:
        feed = feedparser.parse(rss_feed["url"])
        # print("Type of the feed is: ", type(feed))
        # print("#"*80)
        # print(len(feed))
        # print(feed.keys())
        # print("#"*80)
        # print(len(feed["entries"]))
        print("feed.entries[0]")
        print(feed.entries[0])
        for i in range(number_of_entries_to_add):

            documents.append(Document(page_content = feed.entries[i]["summary"],
                                             metadata={"source_url": feed.entries[i]["href"],
                                                        "title": feed.entries[i]["title"],
                                                        "published":feed.entries[i]["published"]
                                                       }))
            log_dict["[ADDED 1]"] = "ADDED 1 RSS entry with the title: " + feed.entries[i]["title"]

        # print("feed.entries[1]")
        # print(feed.entries[1])
        # log_dict["[ADDED 2]"] = "ADDED 2 RSS entry with the title: " + feed.entries[1]["title"]
        # entries_to_documents.append(feed.entries[1]["summary"])

        if feed.status == 200:
            # for entry in feed.entries:
            #     # print(entry.title)
            #     # print(entry.link)
            #     # print("entry type: ", type(entry))
            #     # print("keys: ", entry.keys())
            #     # print(entry.summary)
            #     # print("#"*80)
            #     # print("Now summary_detail")
            #     # print(entry.summary_detail)
            #     pass
            # print(rss_feed["source"], " successfully responded")
            # print(len(feed.entries))
            embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"
            encode_kwargs_normalize_embeddings = True
            log_dict["[CONFIGURATION]: EMBEDDINGS_MODEL_NAME"] = embeddings_model_name
            log_dict["[CONFIGURATION]: ENCODE_KWARGS_NORMALIZE_EMBEDDINGS"] = encode_kwargs_normalize_embeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=embeddings_model_name,
                encode_kwargs={"normalize_embeddings": encode_kwargs_normalize_embeddings},
            )
            print("The embeddings model is: ", embeddings)
            log_dict[f"[CONFIGURATION EMBEDDING MODEL]"] = f"Successful for the type: {type(embeddings)}" 

            vector_1 = embeddings.embed_query(documents[0].page_content)
            vector_2 = embeddings.embed_query(documents[1].page_content)

            assert len(vector_1) == len(vector_2)
            print(f"Generated vectors of length {len(vector_1)}\n")
            print(vector_1[:10])


        else:
            print("Failed to get RSS feed of ", rss_feed["source"], ". Status code:", feed.status)
            failed_rss.append(rss_feed["source"])
            log_dict["FAILED 1"] = "FAILED to process "+ rss_feed["source"]
            print(log_dict)
    # print("Failed rss feeds in total: ", len(failed_rss))
    print("documents")
    print(documents)


if __name__ == "__main__":
    main()
