from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pymongo import MongoClient
import tempfile
import os
import shutil
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
import chromadb
from langchain.chains import VectorDBQA

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust based on your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Chroma client initialization
chroma_client = chromadb.PersistentClient(path="/home/sai/Documents/Saikumar/API/vectordb")

# MongoDB Atlas connection string
mongo_uri = "mongodb+srv://sruthiyadavb99:aOXpXYfqCi0mg75c@db.ndaoqrm.mongodb.net/my_database?retryWrites=true&w=majority&appName=db"

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client.get_database("my_database")
collection = db.get_collection("queries")

llm = LlamaCpp(
    model_path="/home/sai/Downloads/mistral-7b-instruct-v0.2.Q3_K_M.gguf",
    n_gpu_layers=-1,
    n_batch=2048,
    n_ctx=2048,
    f16_kv=True,
    verbose=True,
)

class QueryModel(BaseModel):
    query: str
    prompt: str
    response: str = None
    parsed_at: datetime = None

@app.get("/response/")
async def read_item(prompt: str):
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")  # Updated model name

        # Initialize Chroma retriever correctly
        retriever = Chroma(
            collection_name="my_collection",
            embedding_function=embedding_function,
            client=chroma_client
        )

        # Process query and get response
        qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=retriever)
    
        response = qa.run(prompt)

        # Store query data in the queries collection
        query_data = {
            "query": prompt,
            "response": response,
            "parsed_at": datetime.utcnow()
        }
        collection.insert_one(query_data)

        # Return response
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_files/")
async def get_files():
    try:
        collection = chroma_client.get_collection(name="my_collection")
        files = collection.get()["source"]
        return {"files": files}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/get_col/")
async def get_collection():
    try:
        collections = chroma_client.list_collections()
        return {"collections": collections}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

parsed_data_collection = db.get_collection("parsed_data")

@app.post("/add_pdf/")
async def add_doc(document: UploadFile = File(...)):
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")  # Updated model name
        file_content = await document.read()
        
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
            collection_name="my_collection"
        )
        
        parsed_data = {
            "filename": document.filename,
            "parsed_at": datetime.utcnow(),
            "collection_name": "my_collection",
        }
        collection.insert_one(parsed_data)
        
        os.remove(temp_file_path)
        
        return JSONResponse(content={"response": "Collection and documents added successfully"})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/add_doc/")
async def add_doc(document: UploadFile = File(...)):
    try:
        file_content = await document.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(temp_file_path, temp_dir)

            loader = DirectoryLoader(temp_dir)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
            all_splits = text_splitter.split_documents(documents)

            embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")  # Updated model name
            Chroma.from_documents(
                documents=all_splits,
                embedding_function=embedding_function,
                client=chroma_client,
                collection_name="my_collection"
            )

        os.remove(temp_file_path)

        return JSONResponse(content={"response": "Collection and documents added successfully"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/collection/")
async def get_collection():
    try:
        collection = chroma_client.get_collection("my_collection")
        if not collection:
            return JSONResponse(status_code=404, content={"error": "not found"})
        return {"collection_name": "my_collection", "data": collection.get()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
