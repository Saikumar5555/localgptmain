import pymongo.errors
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient

app = FastAPI()

# MongoDB Atlas connection string
mongo_uri = "mongodb+srv://sruthiyadavb99:aOXpXYfqCi0mg75c@db.ndaoqrm.mongodb.net/my_database?retryWrites=true&w=majority&appName=db"

def connect_to_mongodb(uri, max_retries=3, retry_delay=5):
    attempt = 0
    while attempt < max_retries:
        try:
            client = MongoClient(uri)
            db = client.get_database("my_database")  # Replace with your database name
            return db
        except pymongo.errors.ConnectionFailure as e:
            attempt += 1
            if attempt == max_retries:
                raise HTTPException(status_code=500, detail=f"Failed to connect to MongoDB after {max_retries} attempts: {str(e)}")
            else:
                time.sleep(retry_delay)

@app.on_event("startup")
async def startup_event():
    # Attempt to connect to MongoDB on startup
    app.mongodb_client = connect_to_mongodb(mongo_uri)

@app.get("/test_query/")
async def test_query():
    try:
        # Example query to test MongoDB connection
        result = app.mongodb_client.collection.find_one({})
        return {"message": "MongoDB query successful", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
