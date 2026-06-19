# adapted from
# https://onnyunhui.medium.com/building-a-basic-llm-chat-app-with-streamlit-chat-element-functions-using-only-google-colab-70ab2ce05142
import streamlit as st
from src.main import get_snippets

from pathlib import Path

# from session_state import initialise_session_state, disable_chat_input

def initialise_session_state():
    """
    Initialise session state variables from streamlit
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

def disable_chat_input():
    """
    Disable chat input from streamlit
    """
    st.session_state.processing = True

def normalize_snippet(result):
    """
    Handles both:
    - Document
    - (Document, score)
    """
    if isinstance(result, tuple):
        document, score = result
    else:
        document, score = result, None

    metadata = document.metadata or {}

    return {
        "content": document.page_content,
        "source": metadata.get("source", "unknown source"),
        "page": metadata.get("page", None),
        "score": score,
    }


def render_snippet(snippet, index):
    source_name = Path(snippet["source"]).name
    page = snippet["page"]
    score = snippet["score"]

    with st.container(border=True):
        st.markdown(f"### Snippet {index}")

        meta_parts = [f"**Source:** `{source_name}`"]

        if page is not None:
            meta_parts.append(f"**Page:** {page}")

        if score is not None:
            meta_parts.append(f"**Score:** {score:.4f}")

        st.caption(" · ".join(meta_parts))

        text = snippet["content"].replace("\n", "\n\n")
        st.markdown(text)


# Initialise session state and run the app
st.title("Retriever of book snippets")

initialise_session_state()

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and isinstance(message["content"], list):
            for i, snippet in enumerate(message["content"], start=1):
                render_snippet(snippet, i)
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about Scalability Engineering", disabled=st.session_state.processing, on_submit=disable_chat_input):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.spinner("AI is thinking..."):
        # full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        ai_response = get_snippets(prompt,k=1)
        print(ai_response)

    # Display AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching relevant snippets..."):
            raw_results = get_snippets(prompt, k=3)

            snippets = [
                normalize_snippet(result)
                for result in raw_results
            ]

            for i, snippet in enumerate(snippets, start=1):
                render_snippet(snippet, i)
        
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": snippets})
    
    # Re-enable chat input
    st.session_state.processing = False

    # Rerun the app to update the chat history
    st.rerun()

# to run execute from the root directory
# uv run python -m streamlit run src/streamlit_app.py


# TODO: make it run without installing torchvision
# When I started using streamlit, the main function stopped working because i didn't have torchvision installed
# torchvision is not needed but some dependencies break because they do not see this dependency. This is explanation from GPT:
# So the chain is roughly:

# Streamlit app
#   -> imports src.main
#   -> imports embeddings / sentence-transformers / transformers
#   -> Streamlit file watcher inspects loaded modules
#   -> Transformers lazy image-processing modules get touched
#   -> some image modules expect torchvision

# Hugging Face explicitly treats some dependencies as “soft dependencies”: not everyone needs them, but specific objects do. Their docs say that fast image processors require vision, torch, and torchvision, and that missing torchvision makes fast image processors unavailable.

# For your project, because you are doing text embeddings, not image processing, I would not immediately install torchvision. I would first disable Streamlit’s file watcher.

# Create this file:

# .streamlit/config.toml

# with:

# [server]
# fileWatcherType = "none"

# Streamlit documents fileWatcherType = "none" as the option that turns off file watching completely.

# Then run again:

# uv run streamlit run src/streamlit_app.py

# The downside is that Streamlit will no longer automatically rerun when you save files. You can refresh manually.

# Alternative solution: install torchvision:

# uv add torchvision

# But for your use case, that is probably unnecessary extra dependency weight unless something else actually uses vision models.

# I would try this order:

# 1. Disable Streamlit file watcher with .streamlit/config.toml
# 2. Rerun the app
# 3. Only install torchvision if the app still crashes