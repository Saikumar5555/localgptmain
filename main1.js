const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

// Replace with your FastAPI base URL
const FASTAPI_BASE_URL = 'http://localhost:8000';

// Function to start a new session
async function startSession(collectionName) {
    try {
        const response = await axios.post(`${FASTAPI_BASE_URL}/start_session`, null, {
            params: { collection_name: collectionName }
        });
        console.log('Start Session Response:', response.data);
    } catch (error) {
        console.error('Error starting session:', error);
    }
}

// Function to get chat history
async function getChatHistory(sessionId) {
    try {
        const response = await axios.get(`${FASTAPI_BASE_URL}/chat_history/${sessionId}`);
        console.log('Chat History Response:', response.data);
    } catch (error) {
        console.error('Error getting chat history:', error);
    }
}

// Function to get response
async function getResponse(prompt, collectionName, sessionId) {
    try {
        const response = await axios.get(`${FASTAPI_BASE_URL}/response`, {
            params: {
                prompt: prompt,
                collection_name: collectionName,
                session_id: sessionId
            }
        });
        console.log('Response:', response.data);
    } catch (error) {
        console.error('Error getting response:', error);
    }
}

// Function to get files
async function getFiles(collectionName) {
    try {
        const response = await axios.get(`${FASTAPI_BASE_URL}/get_files`, {
            params: { collection_name: collectionName }
        });
        console.log('Get Files Response:', response.data);
    } catch (error) {
        console.error('Error getting files:', error);
    }
}

// Function to create a collection
async function createCollection(colName) {
    try {
        const response = await axios.get(`${FASTAPI_BASE_URL}/create_col`, {
            params: { col_name: colName }
        });
        console.log('Create Collection Response:', response.data);
    } catch (error) {
        console.error('Error creating collection:', error);
    }
}

// Function to get all collections
async function getCollections() {
    try {
        const response = await axios.get(`${FASTAPI_BASE_URL}/get_col`);
        console.log('Get Collections Response:', response.data);
    } catch (error) {
        console.error('Error getting collections:', error);
    }
}

// Function to add a PDF document
async function addPdf(collectionName, filePath) {
    try {
        const formData = new FormData();
        formData.append('collection_name', collectionName);
        formData.append('document', fs.createReadStream(filePath));

        const response = await axios.post(`${FASTAPI_BASE_URL}/add_pdf`, formData, {
            headers: {
                ...formData.getHeaders()
            }
        });
        console.log('Add PDF Response:', response.data);
    } catch (error) {
        console.error('Error adding PDF:', error);
    }
}

// Function to add a document (zip)
async function addDocument(collectionName, filePath) {
    try {
        const formData = new FormData();
        formData.append('collection_name', collectionName);
        formData.append('document', fs.createReadStream(filePath));

        const response = await axios.post(`${FASTAPI_BASE_URL}/add_doc`, formData, {
            headers: {
                ...formData.getHeaders()
            }
        });
        console.log('Add Document Response:', response.data);
    } catch (error) {
        console.error('Error adding document:', error);
    }
}

// Example usage
(async () => {
    // Replace with appropriate values
    const collectionName = 'example_collection';
    const sessionId = 'example_session_id';
    const prompt = 'example_prompt';
    const pdfFilePath = 'path/to/your/file.pdf';
    const docFilePath = 'path/to/your/file.zip';

    await startSession(collectionName);
    await getChatHistory(sessionId);
    await getResponse(prompt, collectionName, sessionId);
    await getFiles(collectionName);
    await createCollection(collectionName);
    await getCollections();
    await addPdf(collectionName, pdfFilePath);
    await addDocument(collectionName, docFilePath);
})();
