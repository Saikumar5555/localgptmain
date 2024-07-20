from fastapi import FastAPI, File, UploadFile
from chroma import process_document  # Import your document processing function
from pymongo import MongoClient
from pydantic import BaseModel
from chroma import process_document

# Initialize FastAPI app
app = FastAPI()

# MongoDB connection setup
client = MongoClient("your_mongodb_url")
db = client.get_database()
collection = db['queries']

# Define models for request and response
class Document(BaseModel):
    filename: str
    content: str

# Endpoint to upload and process documents
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    # Process document using Chroma or any processing logic
    processed_data = process_document(content)
    # Save processed data to MongoDB
    result = collection.insert_one({
        'filename': file.filename,
        'content': processed_data
    })
    return {"filename": file.filename, "message": "Document processed and saved successfully"}

# Additional endpoints for fetching and managing documents can be added here
