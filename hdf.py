import os
import tempfile
from datetime import datetime

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# MongoDB Atlas connection string
mongo_uri = "mongodb+srv://sruthiyadavb99:aOXpXYfqCi0mg75c@db.ndaoqrm.mongodb.net/my_database?retryWrites=true&w=majority&appName=db"

# Create a FastAPI instance
app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost",
    "http://localhost:8000",
    # Add other origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client.get_database("my_database")  # Replace with your database name
collection = db.get_collection("queries")

# Chroma client (assuming chromadb is already configured and imported)
chroma_client = None  # Replace with actual client if required

@app.post("/add_pdf/")
async def add_doc(collection_name: str = Form(...), documents: list[UploadFile] = File(...)):
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")

        for document in documents:
            file_content = await document.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            loader = PyPDFLoader(temp_file_path)
            pdf_documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
            all_splits = text_splitter.split_documents(pdf_documents)

            Chroma.from_documents(
                documents=all_splits,
                embedding=embedding_function,
                client=chroma_client,
                collection_name=collection_name
            )

            # Save parsed data to MongoDB
            parsed_data = {
                "filename": document.filename,
                "parsed_at": datetime.utcnow(),
                "collection_name": collection_name,
            }
            collection.insert_one(parsed_data)

            os.remove(temp_file_path)

        return JSONResponse(content={"response": "Collection and documents added successfully"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
