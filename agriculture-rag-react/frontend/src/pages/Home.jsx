import React from 'react';
import ChatInterface from '../components/ChatInterface';

const Home = () => {
  return (
    <div className="app-container">
      <div className="sidebar">
        <h2>🌾 Agriculture Document Analysis</h2>
        <p>
          Upload agricultural documents (PDFs or images) and interact with them using AI.
          The system supports:
        </p>
        <ul>
          <li>Text extraction and analysis</li>
          <li>Image understanding</li>
          <li>Question answering</li>
          <li>Document search</li>
        </ul>
      </div>
      <div className="main-content">
        <ChatInterface />
      </div>
    </div>
  );
};

export default Home;