from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List
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

app = FastAPI()
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
    n_ctx=8192,
    f16_kv=True,
    verbose=True,
)

class QueryModel(BaseModel):
    query: str
    chat_history: List[str]
    prompt: str
    collection_name: str
    response: str = None
    parsed_at: datetime = None

@app.get("/insert_data")
def insert_data():
    data = {"key": "prompt"}
    result = collection.insert_one(data)
    return {"message": f"Inserted document ID: {result.inserted_id}"}

@app.get("/response/")
async def read_item(prompt: str, collection_name: str):
    try:
        # Define embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")

        # Initialize chat history as an empty list
        chat_history = []

        # Retrieve existing chat history if available
        existing_data = collection.find_one({"collection_name": collection_name})
        if existing_data and "chat_history" in existing_data:
            chat_history = existing_data["chat_history"]

        # Process query and get response
        retriever = Chroma(collection_name=collection_name, embedding_function=embedding_function)
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )
        response = qa.run(prompt)

        # Append current interaction to chat history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": response})

        # Prepare query data to insert into MongoDB
        query_data = {
            "query": prompt,
            "response": response,
            "chat_history": chat_history,
            "collection_name": collection_name,
            "parsed_at": datetime.utcnow()
        }

        # Insert or update query data into MongoDB
        collection.update_one(
            {"collection_name": collection_name},
            {"$set": query_data},
            upsert=True
        )

        # Return response and updated chat history
        return {"response": response, "chat_history": chat_history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_files/")
async def get_files(collection_name: str):
    try:
        collection = chroma_client.get_collection(name=collection_name)
        files = collection.get()["source"]
        return {"files": files}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/create_col/")
async def create_collection(col_name: str):
    try:
        collection = chroma_client.create_collection(name=col_name)
        return {"response": "collection created"}
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
async def add_doc(collection_name: str = Form(...), document: UploadFile = File(...)):
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")
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
            collection_name=collection_name
        )
        
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

@app.post("/add_doc/")
async def add_doc(collection_name: str = Form(...), document: UploadFile = File(...)):
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

            embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")
            Chroma.from_documents(
                documents=all_splits,
                embedding=embedding_function,
                client=chroma_client,
                collection_name=collection_name
            )

        os.remove(temp_file_path)

        return JSONResponse(content={"response": "Collection and documents added successfully"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/collection/{collection_name}")
async def get_collection(collection_name: str):
    try:
        collection = chroma_client.get_collection(collection_name)
        if not collection:
            return JSONResponse(status_code=404, content={"error": "not found"})
        return {"collection_name": collection_name, "data": collection.get()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
