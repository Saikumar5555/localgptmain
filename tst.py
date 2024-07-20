from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
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
import uuid
import logging

app = FastAPI()
chroma_client = chromadb.PersistentClient(path="/home/sai/Documents/Saikumar/API/vectordb")

# MongoDB Atlas connection string
mongo_uri = "mongodb+srv://sruthiyadavb99:aOXpXYfqCi0mg75c@db.ndaoqrm.mongodb.net/my_database?retryWrites=true&w=majority&appName=db"

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client.get_database("my_database")
collection = db.get_collection("queries")
sessions_collection = db.get_collection("chat_sessions")

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
    prompt: str
    collection_name: str
    session_id: str = None
    response: str = None
    parsed_at: datetime = None

def generate_session_id():
    return str(uuid.uuid4())

@app.post("/start_session")
async def start_session(collection_name: str):
    try:
        session_id = generate_session_id()
        
        session_data = {
            "collection_name": collection_name,
            "session_id": session_id,
            "chat_history": [],
            "created_at": datetime.utcnow()
        }
        
        sessions_collection.insert_one(session_data)
        
        return {"session_id": session_id, "message": "New chat session started"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_history/{session_id}")
async def get_chat_history(session_id: str):
    try:
        chat_data = sessions_collection.find_one({"session_id": session_id})
        
        if chat_data and "chat_history" in chat_data:
            return {"chat_history": chat_data["chat_history"]}
        else:
            return {"message": "No chat history found for this session"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/response/")
async def read_item(prompt: str, collection_name: str, session_id: str = Query(default_factory=generate_session_id)):
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")

        # Retrieve existing chat history if available
        session_data = sessions_collection.find_one({"session_id": session_id})
        if session_data and "chat_history" in session_data:
            chat_history = session_data["chat_history"]
        else:
            chat_history = []

        # Initialize Chroma retriever correctly
        retriever = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            client=chroma_client
        ).as_retriever()

        # Process query and get response
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )
        response = qa.run({"question": prompt, "chat_history": chat_history})

        # Append current interaction to chat history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": response})

        # Update chat history in MongoDB
        sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"chat_history": chat_history}},
            upsert=True
        )

        # Store query data in the queries collection
        query_data = {
            "query": prompt,
            "response": response,
            "session_id": session_id,
            "collection_name": collection_name,
            "parsed_at": datetime.utcnow()
        }
        collection.insert_one(query_data)

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
