import React, { useState } from 'react';
import axios from 'axios';

function App() {
    const [prompt, setPrompt] = useState('');
    const [response, setResponse] = useState('');
    const [chatHistory, setChatHistory] = useState([]);

    const fetchResponse = async () => {
        try {
            // Call /response/ endpoint with prompt
            const response = await axios.get('http://localhost:8000/response/', {
                params: {
                    prompt: prompt,
                    collection_name: 'default', // Assuming a default collection name for now
                    session_id: 'default' // Assuming a default session ID for now
                }
            });

            // Update state with response and chat history
            setResponse(response.data.response);
            setChatHistory(response.data.chat_history);
        } catch (error) {
            console.error('Error fetching response:', error);
        }
    };

    return (
        <div className="App">
            <h1>FastAPI React Frontend</h1>
            <div>
                <label>Prompt:</label>
                <input type="text" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            </div>
            <button onClick={fetchResponse}>Fetch Response</button>
            <div>
                <h2>Response:</h2>
                <p>{response}</p>
            </div>
            <div>
                <h2>Chat History:</h2>
                <ul>
                    {chatHistory.map((item, index) => (
                        <li key={index}>
                            <strong>{item.role}: </strong>
                            {item.content}
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
}

export default App;
