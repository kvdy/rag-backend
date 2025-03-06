from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from typing import List
from dotenv import load_dotenv
import shutil
import os
import psycopg2
from psycopg2.extras import Json
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow requests from frontend (Angular running on localhost:4200)
origins = [
    "http://localhost:4200",  # Allow frontend during development
    "http://127.0.0.1:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows only specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Default Directory to store uploaded documents
DEFAULT_UPLOAD_DIR = "uploaded_docs"
os.makedirs(DEFAULT_UPLOAD_DIR, exist_ok=True)

class IngestRequest(BaseModel):
  directory: str = Form(DEFAULT_UPLOAD_DIR)

class QueryRequest(BaseModel):
    question: str
    directory: str = Form(DEFAULT_UPLOAD_DIR)

class IngestionStatus:
    total_files: int = 0
    processed_files: int = 0
    status: str = "Not Started"

ingestion_status = IngestionStatus()

# Read API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# Initialize embedding model and vector database
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    
# Database Configuration
DB_CONFIG = {
    "dbname": "neondb",
    "user": "neondb_owner",
    "password": "npg_5YTIG2BilcSg",
    "host": "ep-frosty-glitter-a5flpvlc-pooler.us-east-2.aws.neon.tech",
    "port": 5432
}

# Connect to the database
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

vector_db = None

# Load and process document
def load_document(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format")
    return loader.load()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), directory: str = Form(DEFAULT_UPLOAD_DIR)):
    print(directory)
    os.makedirs(directory, exist_ok=True)
    file_location = os.path.join(directory, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "directory": directory, "message": "File uploaded successfully"}


@app.post("/ingest/")
async def ingest_documents(request: IngestRequest):
    print(request)
    directory = request.directory if request.directory else DEFAULT_UPLOAD_DIR
    print(directory)
    global ingestion_status
    os.makedirs(directory, exist_ok=True)
    ingestion_status.status = "In Progress"
    ingestion_status.total_files = len(os.listdir(directory))
    ingestion_status.processed_files = 0
    
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            docs = load_document(file_path)
            for doc in docs:
                doc.metadata["filename"] = filename
            documents.extend(docs)
        except ValueError as e:
            continue
        
        ingestion_status.processed_files += 1
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            filename TEXT,
            directory TEXT,
            text TEXT,
            embedding VECTOR(1536)
        )
    """)
    
    for doc in split_docs:
        embedding = embedding_model.embed_query(doc.page_content)
        cur.execute("INSERT INTO embeddings (filename, directory, text, embedding) VALUES (%s, %s, %s, %s)", 
                    (doc.metadata.get("filename", "unknown"), directory, doc.page_content, Json(embedding)))
    
    conn.commit()
    cur.close()
    conn.close()
    
    ingestion_status.status = "Completed"
    return {"message": "Documents ingested and stored in PostgreSQL successfully", "directory": directory}

@app.get("/ingest/status/")
async def get_ingestion_status():
    return {
        "status": ingestion_status.status,
        "total_files": ingestion_status.total_files,
        "processed_files": ingestion_status.processed_files
    }

@app.post("/query/")
async def query_rag(request: QueryRequest):
    question = request.question
    directory = request.directory if request.directory else DEFAULT_UPLOAD_DIR
    
    conn = get_db_connection()
    cur = conn.cursor()
    question_embedding = embedding_model.embed_query(question)
    
    cur.execute("""
        SELECT filename, text FROM embeddings 
        WHERE directory = %s
        ORDER BY embedding <-> %s 
        LIMIT 5;
    """, (directory, Json(question_embedding)))
    
    retrieved_docs = [(row[0], row[1]) for row in cur.fetchall()]
    cur.close()
    conn.close()
    
    context = " ".join([doc[1] for doc in retrieved_docs])
    response = llm.invoke(f"Question: {question}\nContext: {context}")
    
    return {"response": response, "source_files": list(set([doc[0] for doc in retrieved_docs]))}