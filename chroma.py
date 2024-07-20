from fastapi import FastAPI, File, UploadFile, Form , HTTPException
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings,SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp
import csv
import chromadb
from langchain.vectorstores import Chroma
from pydantic import BaseModel
import chromadb.utils.embedding_functions as embedding_functions
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os 
from langchain.chains import VectorDBQA
from langchain_community.document_loaders import DirectoryLoader
import shutil
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid

def generate_session_id():
    return str(uuid.uuid4())


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
chroma_client = chromadb.PersistentClient(path="/home/sai/Documents/Saikumar/API/vectordb")

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
    chat_history: list
    prompt: str
    collection_name: str



# MongoDB Atlas connection string
mongo_uri = "mongodb+srv://sruthiyadavb99:aOXpXYfqCi0mg75c@db.ndaoqrm.mongodb.net/my_database?retryWrites=true&w=majority&appName=db"

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client.get_database("my_database")  # Replace with your database name
collection = db.get_collection("queries")
sessions_collection = db.get_collection("chat_sessions")

# Define FastAPI endpoint to insert data into MongoDB
# @app.get("/insert_data")
# def insert_data():
#     data = {"key": "prompt"}
#     result = collection.insert_one(data)
#     return {"message": f"Inserted document ID: {result.inserted_id}"}

class PromptRequest(BaseModel):
    prompt: str

@app.post("/response_test/")
async def response_test(request: PromptRequest):
    # embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")
    prompt = request.prompt
    response = llm.invoke(prompt)
    return response

@app.get("/response/")
async def read_item(prompt: str, collection_name: str):
    embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")
    qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=Chroma(collection_name=collection_name, embedding_function=embedding_function))
    response = qa.run(prompt)
    return {"response": response}

# @app.post("/response_test/", response_model=dict)
# async def response_test(request: PromptRequest):
#     try:
#         prompt = request.prompt
#         response = llm.invoke(prompt)
#         return {"response": response}
#     except Exception as e:
#         # Handle specific exceptions as needed
#         return JSONResponse(status_code=500, content={"error": f"Error processing prompt: {str(e)}"})

# @app.post("/start_session")
# async def start_session(collection_name: str):
#     try:
#         session_id = generate_session_id()
        
#         session_data = {
#             "collection_name": collection_name,
#             "session_id": session_id,
#             "chat_history": [],
#             "created_at": datetime.utcnow()
#         }
        
        
#         sessions_collection.insert_one(session_data)
        
#         return {"session_id": session_id, "message": "New chat session started"}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))





# @app.get("/get_files/")
# async def get_files(collection_name: str):
#     try:
#         collection = chroma_client.get_collection(name=collection_name)
#         files = collection.get()["source"]
#         return {"files": files}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/create_col/")
async def create_collection():
    try:
        collection_name = "my_collection"
        collection = chroma_client.create_collection(name=collection_name)
        return {"response": "collection created"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# @app.get("/get_col/")
# async def get_collection():
#     try:
#         collections = chroma_client.list_collections()
#         return {"collections": collections}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

# @app.post("/add_pdf/")
# async def add_doc(collection_name: str = Form(...), document: UploadFile = File(...)):
#     try:
#         embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")
#         file_content = await document.read()
        
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#             temp_file.write(file_content)
#             temp_file_path = temp_file.name
        
#         loader = PyPDFLoader(temp_file_path)
#         documents = loader.load()
        
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
#         all_splits = text_splitter.split_documents(documents)
        
#         Chroma.from_documents(
#             documents=all_splits,
#             embedding=embedding_function,
#             client=chroma_client,
#             collection_name=collection_name
#         )
        
#         os.remove(temp_file_path)
        
#         return JSONResponse(content={"response": "Collection and documents added successfully"})
    
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})


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


# @app.post("/add_doc/")
# async def add_doc(collection_name: str = Form(...), document: UploadFile = File(...)):
#     try:
#         file_content = await document.read()
        
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
#             temp_file.write(file_content)
#             temp_file_path = temp_file.name
        
#         with tempfile.TemporaryDirectory() as temp_dir:
#             shutil.unpack_archive(temp_file_path, temp_dir)

#             loader = DirectoryLoader(temp_dir)
#             documents = loader.load()

#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
#             all_splits = text_splitter.split_documents(documents)

#             embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")
#             Chroma.from_documents(
#                 documents=all_splits,
#                 embedding=embedding_function,
#                 client=chroma_client,
#                 collection_name=collection_name
#             )

#         os.remove(temp_file_path)

#         return JSONResponse(content={"response": "Collection and documents added successfully"})

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
    
# @app.post("/query/")
# async def querying_v1(query_data: QueryModel):
#     try:
#         # Extract the collection name from the query data
#         collection_name = query_data.collection_name
#         # Retrieve the collection from Chroma  # Retrieve the collection from Chroma
#         collection = chroma_client.get_collection(name=collection_name)
        
#         if not collection:
#             raise HTTPException(status_code=404, detail=f"Collection {collection_name} does not exist")

#         vectorstore = Chroma(collection_name=collection_name, embedding_function=SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en"))
        
#         memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
        
#         conv_chain = ConversationalRetrievalChain.from_llm(
#             llm=llm,
#             chain_type="stuff",
#             retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
#             memory=memory,
#             combine_docs_chain_kwargs={"prompt": query_data.prompt},
#             output_key='answer',
#             get_chat_history=lambda h: h,
#             verbose=False
#         )

#         response = conv_chain.run(inputs={"prompt": query_data.query, "chat_history": query_data.chat_history})
        
#         return {"response": response}

#     except ValueError as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/collection/{collection_name}")
# async def get_collection(collection_name: str):
#     try:
#         collection = chroma_client.get_collection(collection_name)
#         if not collection:
#             return JSONResponse(status_code=404, content={"error": "not found"})
#         return {"collection_name": collection_name, "data": collection.get()}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)