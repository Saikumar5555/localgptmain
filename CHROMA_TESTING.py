from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.vectorstores import Chroma as ChromaVectorStore
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid
import fitz
import chromadb
from langchain.chains import VectorDBQA  

app = FastAPI()

# CORS middleware for development, adjust origins as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Atlas connection
mongo_uri = "mongodb+srv://sruthiyadavb99:aOXpXYfqCi0mg75c@db.ndaoqrm.mongodb.net/my_database?retryWrites=true&w=majority&appName=db"
client = MongoClient(mongo_uri)
db = client.get_database("my_database")
collection = db.get_collection("documents")

# LangChain setup
chroma_client = chromadb.PersistentClient(path="/home/sai/Documents/Saikumar/API/vectordb")
llm = LlamaCpp(
    model_path="/home/sai/Downloads/mistral-7b-instruct-v0.2.Q3_K_M.gguf",
    n_gpu_layers=-1,
    n_batch=2048,
    n_ctx=8192,
    f16_kv=True,
    verbose=True,
)

# Helper function to validate PDF
def is_valid_pdf(file_content):
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        doc.close()
        return True
    except Exception as e:
        return False

# Endpoint to add PDF documents and handle queries
@app.post("/add_pdf/")
async def add_pdf(query: str = Form(None), collection_name: str = Form(...), document: UploadFile = File(...)):
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")
        file_content = await document.read()

        if not is_valid_pdf(file_content):
            return JSONResponse(status_code=400, content={"error": "Invalid PDF file"})

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)

        Chroma.from_documents(
            documents=all_splits,
            embedding=embedding_function,
            client=chroma_client,
            collection_name=collection_name
        )

        parsed_data = {
            "filename": document.filename,
            "parsed_at": datetime.utcnow(),
            "collection_name": collection_name,
        }
        collection.insert_one(parsed_data)

        # If a query is provided, perform document query
        if query:
            try:
                qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=Chroma(collection_name=collection_name, embedding_function=embedding_function))
                response = qa.run(query)
                return JSONResponse(content={"response": response})
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"Error querying documents: {str(e)}"})
        else:
            return JSONResponse(content={"message": "PDF uploaded and processed successfully"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error processing PDF: {str(e)}"})

# Run the FastAPI application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

