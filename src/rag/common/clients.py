from src.rag.common.config import MODEL, EMBEDING_MODEL, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def get_model() -> BaseChatModel:
    return init_chat_model(MODEL)

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
