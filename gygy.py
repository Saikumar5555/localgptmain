from fastapi import FastAPI
from pymongo import MongoClient

# MongoDB Atlas connection string
mongo_uri = "mongodb+srv://sruthiyadavb99:aOXpXYfqCi0mg75c@db.ndaoqrm.mongodb.net/my_database?retryWrites=true&w=majority&appName=db"

# Create a FastAPI instance
app = FastAPI()

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client.get_database("my_database")  # Replace with your database name
collection = db.get_collection("queries")

# Define FastAPI endpoint to insert data into MongoDB
@app.get("/insert_data")
def insert_data():
    data = {"key": "value"}
    result = collection.insert_one(data)
    return {"message": f"Inserted document ID: {result.inserted_id}"}


