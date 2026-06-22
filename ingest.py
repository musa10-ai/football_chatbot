from langchain_community.document_loaders import PyPDFLoader, TextLoader
# THE FIX: Updated to the modern LangChain text splitters module
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

DATA_PATH = "data"
DB_PATH = "vectordb"

documents = []

for file in os.listdir(DATA_PATH):
    path = os.path.join(DATA_PATH, file)
    if file.endswith(".pdf"):
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    elif file.endswith(".txt"):
        loader = TextLoader(path)
        documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(chunks, embeddings)
db.save_local(DB_PATH)

print("✅ Vector database created")