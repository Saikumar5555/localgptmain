const axios = require('axios');

const fastApiUrl = 'http://localhost:8000';

const sendPrompt = async (prompt) => {
  try {
    const response = await axios.get(`${fastApiUrl}/response/`, {
      params: { prompt },
      headers: {
        'Content-Type': 'application/json',
      },
    });
    console.log('Response from FastAPI:', response.data);
  } catch (error) {
    if (error.response) {
      console.error('Error response:', error.response.data);
    } else if (error.request) {
      console.error('Error request:', error.request);
    } else {
      console.error('Error message:', error.message);
    }
  }
};
sendPrompt("Who are the leadership team of Buno /-H");

