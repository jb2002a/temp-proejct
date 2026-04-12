from src.rag.common.config import BASIC_MODEL, ADVANCED_MODEL, EMBEDING_MODEL, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, GROQ_MODEL
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv(override=True)

def get_model() -> BaseChatModel:
    return init_chat_model(BASIC_MODEL)

def get_advanced_model() -> BaseChatModel:
    return init_chat_model(ADVANCED_MODEL)

def get_groq_model() -> BaseChatModel:
    return ChatGroq(model=GROQ_MODEL)

def get_embed_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDING_MODEL)

def get_vector_store_from_chroma() -> Chroma:
    """
    Get the index from the ChromaDB
    """
    embeddings = get_embed_model()
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

