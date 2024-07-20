from fastapi import FastAPI, File, UploadFile, Form 
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


app = FastAPI()
chroma_client = chromadb.PersistentClient(path="/home/sai/Documents/Saikumar/API/vectordb")

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/home/sai/Downloads/mistral-7b-instruct-v0.2.Q3_K_M.gguf",
    n_gpu_layers=-1,
    n_batch=2048,
    n_ctx=8192,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
)

@app.get("/response/")
async def read_item(prompt: str,collection_name:str):
    embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")
    qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=Chroma(collection_name=collection_name,embedding_function=embedding_function))
    response = qa.run(prompt)
    return { "response": response}


## get list of all file URLs in vector db
@app.get("/get_files/")
async def get_files(collection_name: str):
    collection_name = "test_collection"
    collection = chroma_client.get_collection(name=collection_name)
    files = collection.get()["source"]
    return {"files": files}


@app.get("/create_col/")
async def create_collection(col_name: str):
    collection = chroma_client.create_collection(name=col_name)
    return {"response": "collection created"}

class ProcessCollection(BaseModel):
    collection_name: str

@app.get("/get_col/")
async def get_collection():
    collections = chroma_client.list_collections()
    return {"collections": print(collections)}

@app.post("/add_pdf/")
async def add_doc(collection_name: str = Form(...), document: UploadFile = File(...)):
    try:
        # Process the collection name
        embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")
        # Read the file content
        file_content = await document.read()
        
        # Write the file content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        # Load the document using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)
        
        # Add the split documents to the collection
        Chroma.from_documents(
        documents=all_splits,
        embedding=embedding_function,
        client=chroma_client,
        collection_name=collection_name
        )
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return JSONResponse(content={"response": "Collection and documents added successfully"})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/add_doc/")
async def add_doc(collection_name: str = Form(...), document: UploadFile = File(...)):
    try:
        # Read the file content
        file_content = await document.read()
        
        # Write the file content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        # Create a temporary directory to extract the zip file
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(temp_file_path, temp_dir)

            # Load the documents using DirectoryLoader
            loader = DirectoryLoader(temp_dir)
            documents = loader.load()

            # Split the documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
            all_splits = text_splitter.split_documents(documents)

            # Add the split documents to the collection
            Chroma.from_documents(
                documents=all_splits,
                embedding=embedding_functions,
                client=chroma_client,
                collection_name=collection_name
            )

        # Clean up the temporary zip file
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
