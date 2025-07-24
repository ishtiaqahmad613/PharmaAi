import os
from dotenv import load_dotenv

load_dotenv()  # This loads the API key from your .env file

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load raw PDFs
DATA_PATH = "data/"

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
print("Length of PDF pages:", len(documents))
print("✅ Script executed successfully")



# step 2: create chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks
text_chunks= create_chunks(extracted_data=documents)
print("Length of Text chunks: ", len(text_chunks))
# step 3: create vector Embeddings
def get_embedding_model():
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embedding_model
embedding_model=get_embedding_model()

# step 4: Store embeddingsin FAIS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)