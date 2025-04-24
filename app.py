import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import ChatMemoryBuffer
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# -----------------------------------------------------------------------------
# 1. Load environment variables
# -----------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------------------------------------------------------
# 2. Initialize or cache the Chat Engine
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_chat_engine():
    # Initialize the Groq LLM
    llm = Groq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)

    # Initialize FastEmbed embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # Set global defaults for LlamaIndex
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Connect to a persistent ChromaDB
    db2 = chromadb.PersistentClient(path="./chroma-db2")
    chroma_collection = db2.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Build an index from the existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

    # Create a memory buffer
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    # Create a Chat Engine with a custom system prompt
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
            "You are a bot for insurance information  "
            "you only answer questions strictly related to insurance policies — including types of policies, coverage options, premiums, and claim processes — based on the knowledgebase provided. If a question is not related to insurance or not in the dataset or knowledgebase provided, politely refuse to answer and remind the user that this chatbot is only for insurance-related queries"
        ),
    )
    return chat_engine

chat_engine = init_chat_engine()

# -----------------------------------------------------------------------------
# 3. Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Bot",
    page_icon="💬",
    layout="centered",
)

# -----------------------------------------------------------------------------
# 4. Streamlit UI
# -----------------------------------------------------------------------------
st.title("Insurance Agent")
st.markdown("A Streamlit chatbot powered by Groq + LlamaIndex for LIC insurance policy information")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    # Start with a friendly assistant greeting
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi how can I help you today?"}
    ]

# Display existing messages in a chat format
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Provide a chat input box at the bottom
user_input = st.chat_input("Your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    agent_response = chat_engine.chat(user_input)

    assistant_message = agent_response.response

    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

    st.rerun()