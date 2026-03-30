import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.text_splitter = SentenceSplitter(chunk_size=1024)

