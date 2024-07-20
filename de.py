from flask import Flask, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
from flask_pymongo import PyMongo
from pymongo import MongoClient
from datetime import datetime
import tempfile
import os
import shutil
import uuid
import logging
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB Atlas connection string
app.config["MONGO_URI"] = "mongodb+srv://sruthiyadavb99:aOXpXYfqCi0mg75c@db.ndaoqrm.mongodb.net/my_database?retryWrites=true&w=majority&appName=db"
mongo = PyMongo(app)
collection = mongo.db.queries
sessions_collection = mongo.db.chat_sessions

# Chroma client setup (assuming setup outside Flask for simplicity)
chroma_client = None  # Replace with your Chroma client setup

# LlamaCpp setup
llm = LlamaCpp(
    model_path="/home/sai/Downloads/mistral-7b-instruct-v0.2.Q3_K_M.gguf",
    n_gpu_layers=-1,
    n_batch=2048,
    n_ctx=8192,
    f16_kv=True,
    verbose=True,
)

# Generate session ID
def generate_session_id():
    return str(uuid.uuid4())

@app.route('/start_session', methods=['POST'])
def start_session():
    try:
        collection_name = request.form.get('collection_name')
        session_id = generate_session_id()

        session_data = {
            "collection_name": collection_name,
            "session_id": session_id,
            "chat_history": [],
            "created_at": datetime.utcnow()
        }
        
        sessions_collection.insert_one(session_data)
        
        return jsonify({"session_id": session_id, "message": "New chat session started"})

    except Exception as e:
        logger.error(f"Error starting session: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat_history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    try:
        chat_data = sessions_collection.find_one({"session_id": session_id})
        
        if chat_data and "chat_history" in chat_data:
            return jsonify({"chat_history": chat_data["chat_history"]})
        else:
            return jsonify({"message": "No chat history found for this session"})

    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/response/', methods=['GET'])
def read_item():
    try:
        prompt = request.args.get('prompt')
        collection_name = request.args.get('collection_name')
        session_id = request.args.get('session_id')

        embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en")

        # Retrieve existing chat history if available
        session_data = sessions_collection.find_one({"session_id": session_id})
        chat_history = session_data.get("chat_history", []) if session_data else []

        logger.info(f"Retrieved session_id: {session_id} for collection_name: {collection_name}")
        logger.info(f"Chat history before processing: {chat_history}")

        # Initialize Chroma retriever correctly
        try:
            retriever = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_function,
                client=chroma_client
            ).as_retriever()
        except Exception as e:
            logger.error(f"Error initializing Chroma retriever: {e}")
            return jsonify({"error": f"Error initializing Chroma retriever: {e}"}), 500

        # Process query and get response
        try:
            qa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                chain_type="stuff"
            )

            # Include chat_history and question in the input to the chain
            response = qa.run({"question": prompt, "chat_history": chat_history})
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return jsonify({"error": f"Error processing query: {e}"}), 500

        logger.info(f"Generated response: {response}")

        # Append current interaction to chat history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": response})

        # Update chat history in MongoDB
        sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"chat_history": chat_history}},
            upsert=True
        )

        logger.info(f"Updated chat history: {chat_history}")

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
        return jsonify({"response": response, "chat_history": chat_history})

    except Exception as e:
        logger.error(f"Error processing response: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_files/', methods=['GET'])
def get_files():
    try:
        collection_name = request.args.get('collection_name')
        collection = chroma_client.get_collection(name=collection_name)
        files = collection.get()["source"]
        return jsonify({"files": files})

    except Exception as e:
        logger.error(f"Error getting files: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/create_col/', methods=['GET'])
def create_collection():
    try:
        col_name = request.args.get('col_name')
        collection = chroma_client.create_collection(name=col_name)
        return jsonify({"response": "collection created"})

    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_col/', methods=['GET'])
def get_collection():
    try:
        collections = chroma_client.list_collections()
        return jsonify({"collections": collections})

    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/add_pdf/', methods=['POST'])
def add_pdf():
    try:
        collection_name = request.form.get('collection_name')
        document = request.files['document']
        file_content = document.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        loader = PyPDFLoader(temp_file_path)
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
        
        parsed_data = {
            "filename": secure_filename(document.filename),
            "parsed_at": datetime.utcnow(),
            "collection_name": collection_name,
        }
        collection.insert_one(parsed_data)
        
        os.remove(temp_file_path)
        
        return jsonify({"response": "Collection and documents added successfully"})

    except Exception as e:
        logger.error(f"Error adding PDF document: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/add_doc/', methods=['POST'])
def add_doc():
    try:
        collection_name = request.form.get('collection_name')
        document = request.files['document']
        file_content = document.read()
        
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

        return jsonify({"response": "Collection and documents added successfully"})

    except Exception as e:
        logger.error(f"Error adding document: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/collection/<collection_name>', methods=['GET'])
def get_collection_data(collection_name):
    try:
        collection = chroma_client.get_collection(collection_name)
        if not collection:
            return jsonify({"error": "not found"}), 404
        return jsonify({"collection_name": collection_name, "data": collection.get()})

    except Exception as e:
        logger.error(f"Error getting collection: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
